import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from .mp_encoder import Mp_encoder
from .contrast import Contrast, Local_Global_Contrast
from GCL.augmentors.functional import drop_feature

def dense_to_sparse_x(feat_index, n_node, n_dim):
    return torch.sparse.FloatTensor(feat_index,torch.ones(feat_index.shape[1]).to(feat_index.device),
                                    [n_node, n_dim])


class Surrogate(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, feat_mask_ratio, edge_mask_ratio):
        super(Surrogate, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop, feats_dim_list[0])
        self.target_mp = Mp_encoder(P, hidden_dim, attn_drop, feats_dim_list[0])
        self.contrast = Contrast(hidden_dim, tau, lam)
        self.LG_contrast = Local_Global_Contrast(hidden_dim, tau, lam)
        self.feat_mask_ratio = feat_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        self.cluter = nn.Sequential(nn.Linear(hidden_dim, 20), nn.Softmax())
        self.mlp_edge_model = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, 1))

    def get_edge_ind_dict(self, g, canonical_etypes):
        etypes = [i[1] for i in canonical_etypes]
        homo_g = dgl.to_homogeneous(g)
        ndata_Type = homo_g.ndata['_TYPE']  # homo_g.ndata['_ID'],
        etype_ntype = {nt[1]: (nt[0], nt[2]) for i, nt in enumerate(canonical_etypes)}
        ntype_count = torch.bincount(ndata_Type)
        ntype_ind = torch.cat([torch.zeros([1], dtype=int).cuda(), torch.cumsum(ntype_count, 0)], dim=0).cpu().tolist()
        n_ind_dict = {nt: (ntype_ind[i], ntype_ind[i + 1]) for i, nt in enumerate(g.ntypes)}
        e_ind_dict = {nt: (n_ind_dict[etype_ntype[nt][0]], n_ind_dict[etype_ntype[nt][1]]) for nt in etypes}
        return e_ind_dict

    def mask_edge(self, edges, p):
        node_num = edges.shape[0]
        edge_index, weight = edges.indices(), edges.val
        if p < 0. or p > 1.:
            raise ValueError(f'Mask probability has to be between 0 and 1 '
                             f'(got {p}')
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, p, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        edge_index1, edge_weight1 = edge_index[:, ~mask], weight[~mask]
        aug_g = torch.sparse_coo_tensor(edge_index1, edge_weight1, (node_num, node_num)).to_dense().to(edge_index.device)
        return aug_g

    def mask_feature(self, x, p=0.1):
        # x = copy.deepcopy(h)
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(p * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        return out_x


    def get_aug_metapaths(self, adj, edges_id, sp=False):
        aug_edges = [adj[i[0]:i[1], i[2]:i[3]] for i in edges_id]
        # 生成元路径
        aug_mps = []
        for edges in aug_edges:
            mp = edges @ edges.T
            if sp == True:
                aug_mps.append(mp.to_sparse())
            else:
                aug_mps.append(mp)
        return aug_mps

    def get_subg(self, adj, edges_id, sp=False):
        aug_edges = [adj[i[0]:i[1], i[2]:i[3]] for i in edges_id]
        edges_list = []
        for edges in aug_edges:
            edges_list.append(edges.to_sparse())
        return edges_list

    def get_metapaths(self, adj1_list, sp=False):
        metapath_list = []
        for adj in adj1_list:
            metapath = torch.matmul(adj, adj.T)
            metapath = metapath.cuda()
            if sp == True:
                metapath_list.append(metapath.to_sparse())
            else:
                metapath_list.append(metapath)
        return metapath_list

    def forward(self, feats, pos, mps, ori_g, edges_id):  # p a s
        device = feats[0].device
        paper_num = feats[0].shape[0]
        node_num = ori_g.shape[0]
        h_all, aug_mps = [], []
        for i in range(len(feats)):
            # h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
            h_all.append(F.elu(self.fc_list[i](feats[i])))
        # 原始图增广
        edges, weight = ori_g._indices(), ori_g._values()
        edge_index1, edge_weight1 = self.mask_edge(edge_index=edges, weight=weight, p=self.edge_mask_ratio)
        aug_g = torch.sparse_coo_tensor(edge_index1, edge_weight1, (node_num, node_num)).to_dense().to(device)
        aug_edges = [aug_g[i[0]:i[1], i[2]:i[3]] for i in edges_id]
        # 生成元路径
        aug_mps = []
        for edges in aug_edges:
            mp = edges @ edges.T
            # mp = torch.clamp(mp, 0, 1)
            aug_mps.append(mp)
        # 特征扰动
        aug_feat = drop_feature(h_all[0], self.feat_mask_ratio)

        z_mp = self.mp(h_all[0], mps)
        z_mp2 = self.mp(aug_feat, aug_mps)
        loss = self.contrast(z_mp, z_mp2, pos)
        return loss

    def get_embeds(self, feats, mps, fc=False, detach=True):
        if fc == True:
            z_mp = feats[0]
        else:
            z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps, fc=True)

        if detach == True: return z_mp.detach()
        else: return z_mp

class Edge_learnable(nn.Module):
    def __init__(self, hidden_dim, K, top_k, e_type):
        super(Edge_learnable, self).__init__()
        self.hidden_dim = hidden_dim
        self.K = K
        self.top_k = top_k
        self.e_type = e_type
        self.add_edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, 1))

        self.drop_edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, 1))

        self.add_edge_mlp_dict = nn.ModuleDict({i: self.add_edge_mlp for i in self.e_type})
        self.drop_edge_mlp_dict = nn.ModuleDict({i: self.drop_edge_mlp for i in self.e_type})

        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())

        self.cluster = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.Linear(hidden_dim, self.K))

        self.project_intra = torch.nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, hidden_dim))

        self.feat_trans = nn.ModuleDict({i: nn.Linear(hidden_dim, hidden_dim, bias=True) for i in self.e_type})

    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (x * y).sum(dim=-1).pow_(alpha)
        loss = loss.mean()
        return loss

    def cluster_transfer_kl(self, ori_embedding, adv_embedding):
        ori_logit = self.cluster(ori_embedding)
        adv_logit = self.cluster(adv_embedding)
        ori_clusters = F.softmax(ori_logit)
        adv_clusters = F.softmax(adv_logit)
        loss = -F.kl_div(adv_clusters.log(), ori_clusters)
        return loss

    def cluster_transfer_ce(self, ori_embedding, adv_embedding):
        ori_logit = self.cluster(ori_embedding)
        adv_logit = self.cluster(adv_embedding)
        ori_clusters = F.softmax(ori_logit)
        adv_clusters = F.softmax(adv_logit)
        loss = -F.nll_loss(input=adv_clusters, target=torch.argmax(ori_clusters, dim=1))
        return loss

    def forward(self, src_embeddings, adj_list, grad_list, K, e_ind_dict, p, return_metapath=False, drop_fold=2):
        etypes = list(e_ind_dict.keys())
        sub_adj_list, metapaths = [], []
        add_id_list = []
        n = 0
        for e, sub_g, sub_grad in zip(etypes, adj_list, grad_list):
            sub_grad = sub_grad.cuda()
            n_edge = int(sub_g.sum().item())
            n_src, n_dst = sub_g.size(0), sub_g.size(1)

            # dst node obtaining
            dst_embedding = torch.spmm(sub_g.T, F.relu(self.feat_trans[e](src_embeddings)))
            add_edge_num, drop_edge_num = int(p * n_edge), int(p * n_edge)

            if type(self.top_k) == list:
                topk = self.top_k[n]
            elif type(self.top_k) == dict:
                topk = self.top_k[e]
            else:
                if self.top_k < 1:
                    topk = int(add_edge_num * self.top_k)
                else:
                    topk = self.top_k

            exist_grad, non_grad = (sub_grad * sub_g), sub_grad * (1 - sub_g)
            exist_grad_sum_1d, non_grad_sum_1d = exist_grad.reshape(-1), non_grad.reshape(-1)
            exist_values, exist_indices = exist_grad_sum_1d.sort()

            drop_idx = exist_indices[:drop_edge_num * drop_fold]
            drop_idx_dense = torch.stack([drop_idx // n_dst, drop_idx % n_dst])
            drop_src_embedding = src_embeddings[drop_idx_dense[0]]
            drop_dst_embedding = dst_embedding[drop_idx_dense[1]]
            drop_e_embedding = torch.cat([drop_src_embedding, drop_dst_embedding], -1)

            # drop MLP
            drop_edge_logits = self.drop_edge_mlp_dict[e](drop_e_embedding).squeeze(-1)
            selected_src_p = F.gumbel_softmax(drop_edge_logits, tau=1)
            _, topk_idx = torch.topk(selected_src_p, drop_edge_num)#, largest=False
            selected_drop_e_id = torch.zeros_like(selected_src_p)
            selected_drop_e_id = selected_drop_e_id.scatter(-1, topk_idx, 1.)
            selected_drop_e_id = selected_drop_e_id - selected_src_p.detach() + selected_src_p

            drop = torch.zeros(n_src, n_dst).cuda()
            drop[drop_idx_dense[0], drop_idx_dense[1]] = selected_drop_e_id

            add_src_id = drop.nonzero()[:, 0]
            add_dst_id_ = sub_grad[add_src_id].sort(descending=True).indices[:, :topk]

            add_src_embedding = src_embeddings[add_src_id]
            add_dst_embedding = dst_embedding[add_dst_id_]

            add_src_embedding = add_src_embedding.unsqueeze(1).expand_as(add_dst_embedding)
            add_edge_emb = torch.cat([add_src_embedding, add_dst_embedding], -1)

            # add mlp
            add_edge_logits = self.add_edge_mlp_dict[e](add_edge_emb).squeeze(-1)
            selected_dst = F.gumbel_softmax(add_edge_logits, tau=1, hard=True)

            add = torch.zeros(n_src, n_dst).cuda()
            for row in range(add_src_id.shape[0]):
                add[add_src_id[row], add_dst_id_[row]] = selected_dst[row]

            sub_g = sub_g.detach()
            adj_adv = ((sub_g * (1 - drop)) + add * 1.0)

            sub_adj_list.append(adj_adv.to_sparse())
            metapath = torch.matmul(adj_adv, adj_adv.T)
            metapaths.append(metapath.to_sparse())
            add_id_list.append(add_src_id)
            n += 1
        if return_metapath == False:
            return sub_adj_list
        else:
            return sub_adj_list, metapaths, torch.cat(add_id_list).unique()



