import numpy
import numpy as np
import torch
from utils.load_data_DGL import load_data
from utils.params import set_params
from module.HGAC import Surrogate, Edge_learnable
import warnings
import datetime
import random
from torch_geometric.utils import to_dense_adj
from copy import deepcopy
import dgl
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

attack_model = 'HGAC' # SHAC CLGA, variant_cluster

def dense_to_sparse_x(feat_index, n_node, n_dim):
    return torch.sparse.FloatTensor(feat_index, torch.ones(feat_index.shape[1]).to(feat_index.device),
                                    [n_node, n_dim])


def mask_edge(edge_index, p, weight=None):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    elif p == 0:
        return edge_index, weight
    else:
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, p, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        return edge_index[:, ~mask], weight[~mask]


def drop_feature(x: torch.Tensor, p: float) -> torch.Tensor:
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    elif p == 0:
        return x
    else:
        device = x.device
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < p
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0
    return x

def drop_node(x: torch.Tensor, p: float) -> torch.Tensor:
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    elif p == 0:
        return x
    else:
        device = x.device
        drop_mask = torch.empty((x.size(0),), dtype=torch.float32).uniform_(0, 1) < p
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[drop_mask] = 0
    return x

def run_kmeans(x, k):
    estimator = KMeans(n_clusters=k)
    estimator.fit(x)
    y_pred = estimator.predict(x)
    return y_pred

def train():
    If_pre_training = True #True, False

    '''
    所有的数据都首先是个list
    nei_index: list长度是节点类型，内容貌似是，给定每个节点的邻居是哪些， Network Schema 视角
    feats: list长度是节点类型，内容是每个类型的节点特征
    mps: list长度是元路径的类型，内容是每个元路径的edge_index
    pos: 一个正样本的稀疏矩阵，包含了主节点的正样本对，矩阵中如果（a,b）元素为1,则为a,b为正样本对
    label: 标签
    '''
    hg, canonical_etypes, main_type, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num, return_hg=True)
    nb_classes = label.shape[-1]  # 节点类型的个数
    feats_dim_list = [i.shape[1] for i in feats]  # 不同类型节点的维度
    P = int(len(mps))  # 元路径的个数
    print("seed ", args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)
    print(args)
    # HeCo是预训练任务
    model = Surrogate(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                 P, args.sample_rate, args.nei_num, args.tau, args.lam, args.feat_mask_ratio1, args.edge_mask_ratio1)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(device)
        hg = hg.to(device)
        feats = [feat.to(device) for feat in feats]
        mps = [mp.to(device) for mp in mps]
        pos = pos.to(device)

    cnt_wait = 0
    best = 1e9
    best_t = 0
    starttime = datetime.datetime.now()
    K = 5#20
    # grad_list = []
    adj_list = []
    for meta_edge in canonical_etypes:
        src_type, etype, dst_type = meta_edge
        n_src, n_dst = hg.number_of_nodes(src_type), hg.number_of_nodes(dst_type)
        edges = hg.adj(etype=etype)
        # e_index, e_weight = edges.indices(), edges.val   # linux
        e_index, e_weight = edges.coalesce().indices(), edges.coalesce().values()   # windows
        max_num_nodes = max(n_src, n_dst)
        dense_edges = to_dense_adj(e_index, max_num_nodes=max_num_nodes)[0][:n_src, :n_dst].to(device)
        adj_list.append(dense_edges)
        e_ind_dict = model.get_edge_ind_dict(hg, canonical_etypes)

    if If_pre_training == True:
        for epoch in range(args.nb_epochs):
        # for epoch in range(2):
            epoch_loss = 0.0
            '''mini-batch'''
            x = feats[0]
            ori_g = dgl.to_homogeneous(hg).adj().cuda()
            node_num = ori_g.shape[0]

            model.train()
            optimiser.zero_grad()
            # data augmentation
            x1, x2 = x, x
            adj1_list, adj2_list = [], []
            # 获得异构图的原始结构
            for meta_edge in canonical_etypes:
                src_type, etype, dst_type = meta_edge
                n_src, n_dst = hg.number_of_nodes(src_type), hg.number_of_nodes(dst_type)
                edges = hg.adj(etype=etype)
                # e_index, e_weight = edges.indices(), edges.val  # linux
                e_index, e_weight = edges.coalesce().indices(), edges.coalesce().values()  # windows
                edge_index1, edge_weight1 = mask_edge(edge_index=e_index, weight=e_weight, p=args.edge_mask_ratio1)  # edges, weight
                edge_index2, edge_weight2 = mask_edge(edge_index=e_index, weight=e_weight, p=args.edge_mask_ratio2)
                max_num_nodes = max(n_src, n_dst)

                dense_edges1 = to_dense_adj(edge_index1, max_num_nodes=max_num_nodes)[0][:n_src, :n_dst]
                dense_edges2 = to_dense_adj(edge_index2, max_num_nodes=max_num_nodes)[0][:n_src, :n_dst]

                adj1_list.append(dense_edges1)
                adj2_list.append(dense_edges2)
            if epoch > args.warmup:
                for adj1 in adj1_list:
                    adj1.requires_grad_()

                for adj2 in adj2_list:
                    adj2.requires_grad_()

            mps1, mps2 = model.get_metapaths(adj1_list, sp=True), model.get_metapaths(adj2_list, sp=True)

            z1 = model.mp(x1, mps1, fc=True)
            z2 = model.mp(x2, mps2, fc=True)

            loss = model.contrast(z1, z2, pos)
            loss.backward()
            epoch_loss += loss

            # ''' 对抗训练, 对抗视角和增广视角进行对比学习'''
            if epoch > args.warmup:
                grad_list = []
                for adj1, adj2 in zip(adj1_list, adj2_list):
                    adj_grad = adj1.grad + adj2.grad
                    grad_list.append(adj_grad)
            optimiser.step()
            torch.cuda.empty_cache()

            print('Epoch--', epoch, "train loss ", epoch_loss)
            if epoch > args.warmup:
                if epoch_loss < best:
                    best = epoch_loss
                    best_t = epoch
                    best_grad = deepcopy(grad_list)
                    cnt_wait = 0
                    torch.save(model.state_dict(), 'Surrogate_' + own_str + '.pkl')
                else:
                    cnt_wait += 1

                if cnt_wait == args.patience:
                    torch.save(best_grad, own_str + '_best_grad' + '.pt')
                    print('Early stopping!')
                    break
    print('预训练完成')

    '''预训练完成'''
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('Surrogate_' + own_str + '.pkl'))
    model.eval()
    best_grad = torch.load(own_str + '_best_grad' + '.pt')
    print('Loading best grad.')
    if args.dataset == 'freebase':
        top_k = {'ma': 20, 'md': 10, 'mw': 10}
    elif args.dataset == 'acm':
        top_k = {'pa': 15, 'ps': 10} # pa:15, 10
    elif args.dataset == 'aminer':
        top_k = {'pa': 30, 'pr': 30}
    elif args.dataset == 'dblp':
        top_k = {'ap': 30, 'pc': 5, 'pt': 20}
    elif args.dataset == 'imdb':
        top_k = {'ma': 20, 'md': 20}

    fine_model = Edge_learnable(args.hidden_dim, K=20, top_k=top_k, e_type=list(e_ind_dict.keys())).to(device)
    fine_optimiser = torch.optim.Adam(fine_model.parameters(), lr=args.fine_lr, weight_decay=args.l2_coef)
    # K imdb 30
    # lr  acm 1e-4; freebase 1e-4; imdb 1e-5;
    embeds = model.get_embeds(feats, mps, fc=True)

    if attack_model == 'HGAC':
        best = 1e9
        fine_model.train()
        for fine_epoch in range(1200):

            adv_adj, mps_adv, _ = fine_model(embeds, adj_list, best_grad, K, e_ind_dict,
                                          p=args.edge_attack_ratio, return_metapath=True, drop_fold=4)

            embeds_adv = model.get_embeds(feats, mps_adv, fc=True, detach=False)
            fine_loss = fine_model.cluster_transfer_ce(embeds, embeds_adv)

            print('Epoch,', fine_epoch, 'loss', fine_loss)
            fine_optimiser.zero_grad()
            fine_loss.backward()
            fine_optimiser.step()

            if fine_loss < best:
                best = fine_loss
                adv_adj = [i.detach() for i in adv_adj]
                best_adj = deepcopy(adv_adj)
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break

    if args.dataset == 'acm':
        np.savetxt('../data/' + args.dataset + "/" + 'pa_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt', best_adj[0].indices().cpu().numpy())
        np.savetxt('../data/' + args.dataset + "/" + 'ps_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt', best_adj[1].indices().cpu().numpy())
    elif args.dataset == 'freebase':
        np.savetxt('../data/' + args.dataset + "/" + 'ma_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt',  best_adj[0].indices().cpu().numpy())
        np.savetxt('../data/' + args.dataset + "/" + 'md_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt',  best_adj[1].indices().cpu().numpy())
        np.savetxt('../data/' + args.dataset + "/" + 'mw_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt',  best_adj[2].indices().cpu().numpy())
    elif args.dataset == 'aminer':
        np.savetxt('../data/' + args.dataset + "/" + 'pa_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt', best_adj[0].indices().cpu().numpy())
        np.savetxt('../data/' + args.dataset + "/" + 'pr_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt', best_adj[1].indices().cpu().numpy())
    elif args.dataset == 'imdb':
        np.savetxt('../data/' + args.dataset + "/" + 'ma_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt', best_adj[0].indices().cpu().numpy())
        np.savetxt('../data/' + args.dataset + "/" + 'md_' + attack_model + '_gumbel_' + str(args.edge_attack_ratio) + '.txt', best_adj[1].indices().cpu().numpy())
    print('Attacked data loaded！')


if __name__ == '__main__':
    train()
