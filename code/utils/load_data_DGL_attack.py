import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder
import dgl
from torch_geometric.utils import remove_self_loops, segregate_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def comple_isolated_nodes(edge_index):
    edge_index = torch.IntTensor(edge_index)
    src_num = edge_index[0].max().item() + 1
    dst_num = edge_index[1].max().item() + 1
    exist_node = torch.unique(edge_index[1])
    miss_node = torch.IntTensor(get_inverse_id(exist_node, dst_num))

    dst_unif = torch.ones(1, src_num)
    add_idx = dst_unif.multinomial(len(miss_node), replacement=False)
    add_edge_index = torch.cat((add_idx, miss_node.unsqueeze(0)), dim=0)
    edge_index = torch.cat((edge_index, add_edge_index), dim=1)
    return edge_index.cpu().numpy()

def get_inverse_id(index, num):
    return_list = []
    for i in range(num):
        if i not in index:
            return_list.append(i)
    return return_list

def EdgePerturb(edge_index, aug_ratio, src_num, dst_num):
    edge_index = torch.IntTensor(edge_index)
    _, edge_num = edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    src_unif = torch.ones(1, src_num)
    dst_unif = torch.ones(1, dst_num)

    # # 随机抽样
    unif = torch.ones(edge_num)
    remove_idx = unif.multinomial((permute_num), replacement=False)

    # add_src_idx = src_unif.multinomial(permute_num, replacement=True)
    add_src_idx = edge_index[0, remove_idx.unsqueeze(0)]
    add_dst_idx = dst_unif.multinomial(permute_num, replacement=True)
    add_edge_idx = torch.cat([add_src_idx, add_dst_idx], dim=0)
    keep_id = get_inverse_id(remove_idx.tolist(), edge_num)

    edge_index = torch.cat((edge_index[:, keep_id], add_edge_idx), dim=1)
    # edge_index = torch.cat((edge_index, add_edge_idx), dim=1)
    # edge_index = edge_index[:, keep_id]
    return edge_index.cpu().numpy()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(ratio, type_num, return_hg=False, attack_model='CLGA', attack_ratio=0.1):
    # The order of node types: 0 p 1 a 2 s
    # path = "../../data/acm/"
    path = "../data/acm/"
    # import os
    # print(os.path.exists("../data"))
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    P, A, S = 4019, 7167, 60
    pos_num = 5

    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])

    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))

    # 加载原始数据
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)

    def generate_metapaths(pa, ps):
        pa, ps = pa.T, ps.T
        pa_ = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
        ps_ = sp.coo_matrix((np.ones(ps.shape[0]), (ps[:, 0], ps[:, 1])), shape=(P, S)).toarray()

        pap = np.matmul(pa_, pa_.T) > 0
        pap = sp.coo_matrix(pap)
        psp = np.matmul(ps_, ps_.T) > 0
        psp = sp.coo_matrix(psp)
        return pap, psp

    def generate_pos(pap, psp):
        pap = pap / pap.sum(axis=-1).reshape(-1, 1)
        psp = psp / psp.sum(axis=-1).reshape(-1, 1)
        all = (pap + psp).A.astype("float32")

        pos = np.zeros((P, P))
        k = 0
        for i in range(len(all)):
            one = all[i].nonzero()[0]
            if len(one) > pos_num:
                oo = np.argsort(-all[i, one])
                sele = one[oo[:pos_num]]
                pos[i, sele] = 1
                k += 1
            else:
                pos[i, one] = 1
        pos = sp.coo_matrix(pos)
        return pos

    if attack_model == 'random':
        pa = np.loadtxt(path + 'pa.txt', dtype=int).T
        ps = np.loadtxt(path + 'ps.txt', dtype=int).T

        pa = EdgePerturb(pa, attack_ratio, P, A)
        ps = EdgePerturb(ps, attack_ratio, P, S)

        # pa = comple_isolated_nodes(pa)

        pap, psp = generate_metapaths(pa, ps)
        pos = generate_pos(pap, psp)

    elif attack_model == 'CLGA':
        pa = np.loadtxt(path + 'pa_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        ps = np.loadtxt(path + 'ps_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        # pa = comple_isolated_nodes(pa)

        pap, psp = generate_metapaths(pa, ps)
        pos = generate_pos(pap, psp)

    elif attack_model == 'SHAC':
        pa = np.loadtxt(path + 'pa_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        ps = np.loadtxt(path + 'ps_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        # pa = comple_isolated_nodes(pa)

        pap, psp = generate_metapaths(pa, ps)
        pos = generate_pos(pap, psp)

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]

    # 构建原始图
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): (pa[0], pa[1]),
        ('paper', 'ps', 'subject'): (ps[0], ps[1]),
        ('subject', 'sp', 'paper'): (ps[1], ps[0]),
        ('author', 'ap', 'paper'): (pa[1], pa[0])
    })

    hg.nodes['paper'].data['h'] = feat_p
    hg.nodes['author'].data['h'] = feat_a
    hg.nodes['subject'].data['h'] = feat_s

    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('paper', 'pa', 'author'), ('paper', 'ps', 'subject')]
    main_type = 'paper'

    if return_hg == True:
        return hg, canonical_etypes, main_type, [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test
    else:
        return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test

def load_dblp(ratio, type_num, return_hg=False):
    # The order of node types: 0 a 1 p 2 c 3 t
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "../data/dblp/"
    # path = "../../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)

    pa = np.loadtxt(path + 'pa.txt', dtype=int).T
    pc = np.loadtxt(path + 'pc.txt', dtype=int).T
    pt = np.loadtxt(path + 'pt.txt', dtype=int).T
    # p_num, a_num, c_num = 14328, 4057,

    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): (pa[0], pa[1]),
        ('author', 'ap', 'paper'): (pa[1], pa[0]),
        ('paper', 'pc', 'conference'): (pc[0], pc[1]),
        ('conference', 'cp', 'paper'): (pc[1], pc[0]),
        ('paper', 'pt', 'term'): (pt[0], pt[1]),
        ('term', 'pt', 'paper'): (pt[1], pt[0]),
    })
    # hg = dgl.to_bidirected(hg)
    hg.nodes['paper'].data['h'] = feat_p
    hg.nodes['author'].data['h'] = feat_a

    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    canonical_etypes = [('author', 'ap', 'paper'), ('paper', 'pt', 'term'), ('paper', 'pc', 'conference')]
    # metapaths = ['auther-paper-author', 'auther-paper-conference-paper-auther', 'auther-paper-term-paper-auther']
    main_type = 'author'
    if return_hg == True:
        return hg, canonical_etypes, main_type, [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test
    else:
        return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_aminer(ratio, type_num, return_hg=False):
    # The order of node types: 0 p 1 a 2 r
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)

    pa = np.loadtxt(path + 'pa.txt', dtype=int).T
    pr = np.loadtxt(path + 'pr.txt', dtype=int).T
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]  #
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]

    # pa, pr = torch.Tensor(pa), torch.Tensor(pr)
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    # 构建原始图
    # node_num = type_num[0] + type_num[1] + type_num[2]
    # pa[1] = pa[1] + type_num[0]
    # pr[1] = pr[1] + type_num[0] + type_num[1]
    # ori_g = np.concatenate([pa, pr], axis=1)
    # ori_g = th.LongTensor(ori_g)
    # ori_g = th.sparse_coo_tensor(ori_g, th.ones(ori_g.size(1)), (node_num, node_num))#.to_dense()
    # edges_id = [(0, type_num[0], type_num[0], type_num[0] + type_num[1]),
    #            (0, type_num[0], type_num[0] + type_num[1], node_num)]

    #
    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): (pa[0], pa[1]),
        ('author', 'ap', 'paper'): (pa[1], pa[0]),
        ('paper', 'pr', 'ref'): (pr[0], pr[1]),
        ('ref', 'rp', 'paper'): (pr[1], pr[0]),
    })
    # hg = dgl.to_bidirected(hg)
    hg.nodes['paper'].data['h'] = feat_p
    hg.nodes['author'].data['h'] = feat_a
    hg.nodes['ref'].data['h'] = feat_r

    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('paper', 'pa', 'author'), ('paper', 'pr', 'ref')]
    main_type = 'paper'

    if return_hg == True:

        return hg, canonical_etypes, main_type, [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test
    else:
        return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test


def load_freebase(ratio, type_num, return_hg, attack_model):
    # The order of node types: 0 m 1 d 2 a 3 w
    M, D, A, W = 3492, 2502, 33401, 4459
    pos_num = 80
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))

    def generate_metapaths(ma, md, mw):
        ma, md, mw = ma.T, md.T, mw.T
        ma_ = sp.coo_matrix((np.ones(ma.shape[0]), (ma[:, 0], ma[:, 1])), shape=(M, A)).toarray()
        md_ = sp.coo_matrix((np.ones(md.shape[0]), (md[:, 0], md[:, 1])), shape=(M, D)).toarray()
        mw_ = sp.coo_matrix((np.ones(mw.shape[0]), (mw[:, 0], mw[:, 1])), shape=(M, W)).toarray()

        mam = np.matmul(ma_, ma_.T) > 0
        mam = sp.coo_matrix(mam)
        mdm = np.matmul(md_, md_.T) > 0
        mdm = sp.coo_matrix(mdm)
        mwm = np.matmul(mw_, mw_.T) > 0
        mwm = sp.coo_matrix(mwm)
        return mam, mdm, mwm

    def generate_pos(mam, mdm, mwm):
        mam = mam / mam.sum(axis=-1).reshape(-1, 1)
        mdm = mdm / mdm.sum(axis=-1).reshape(-1, 1)
        mwm = mwm / mwm.sum(axis=-1).reshape(-1, 1)
        all = (mam + mdm + mwm).A.astype("float32")

        pos = np.zeros((M, M))
        k = 0
        for i in range(len(all)):
            one = all[i].nonzero()[0]
            if len(one) > pos_num:
                oo = np.argsort(-all[i, one])
                sele = one[oo[:pos_num]]
                pos[i, sele] = 1
                k += 1
            else:
                pos[i, one] = 1
        pos = pos + np.eye(M)
        pos = sp.coo_matrix(pos)
        return pos

    def gen_neibor(edges):
        e_dict = {}
        for i in edges:
            if i[0] not in e_dict:
                e_dict[int(i[0])] = []
                e_dict[int(i[0])].append(int(i[1]))
            else:
                e_dict[int(i[0])].append(int(i[1]))
        e_keys = sorted(e_dict.keys())
        e_nei = [e_dict[i] for i in e_keys]
        e_nei = np.array([np.array(i) for i in e_nei], dtype=object)
        return e_nei

    attack_ratio = 0.1

    if attack_model == 'random':
        ma = np.loadtxt(path + 'ma.txt', dtype=int).T
        md = np.loadtxt(path + 'md.txt', dtype=int).T
        mw = np.loadtxt(path + 'mw.txt', dtype=int).T

        ma = EdgePerturb(ma, attack_ratio, M, A)
        md = EdgePerturb(md, attack_ratio, M, D)
        mw = EdgePerturb(mw, attack_ratio, M, W)

        add = np.array([[3491], [33400]])
        ma = np.concatenate([ma, add], axis=1)

        mam, mdm, mwm = generate_metapaths(ma, md, mw)
        pos = generate_pos(mam, mdm, mwm)

    elif attack_model == 'CLGA':
        ma = np.loadtxt(path + 'ma_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mw = np.loadtxt(path + 'mw_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        add = np.array([[3491], [33400]])
        ma = np.concatenate([ma, add], axis=1)

        mam, mdm, mwm = generate_metapaths(ma, md, mw)
        pos = generate_pos(mam, mdm, mwm)

    elif attack_model == 'SHAC':
        ma = np.loadtxt(path + 'ma_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mw = np.loadtxt(path + 'mw_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        add = np.array([[3491], [33400]])
        ma = np.concatenate([ma, add], axis=1)

        mam, mdm, mwm = generate_metapaths(ma, md, mw)
        pos = generate_pos(mam, mdm, mwm)

    # mam, mdm = generate_metapaths(ma, md)
    # ma = comple_isolated_nodes(ma)
    nei_a = gen_neibor(ma)
    nei_d = gen_neibor(md)
    nei_w = gen_neibor(mw)

    # DGL he graph
    hg = dgl.heterograph({
        ('movie', 'ma', 'actor'): (ma[0], ma[1]),
        ('actor', 'am', 'movie'): (ma[1], ma[0]),
        ('movie', 'md', 'direct'): (md[0], md[1]),
        ('direct', 'dm', 'movie'): (md[1], md[0]),
        ('movie', 'mw', 'writer'): (mw[0], mw[1]),
        ('writer', 'wm', 'movie'): (mw[1], mw[0]),
    })
    # hg = dgl.to_bidirected(hg)
    hg.nodes['movie'].data['h'] = feat_m
    hg.nodes['actor'].data['h'] = feat_a
    hg.nodes['direct'].data['h'] = feat_d
    hg.nodes['writer'].data['h'] = feat_w

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i.astype(float)) for i in nei_d]
    nei_a = [th.LongTensor(i.astype(float)) for i in nei_a]
    nei_w = [th.LongTensor(i.astype(float)) for i in nei_w]

    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('movie', 'ma', 'actor'), ('movie', 'md', 'direct'), ('movie', 'mw', 'writer')]
    main_type = 'movie'
    if return_hg == True:
        return hg, canonical_etypes, main_type, [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test
    else:
        return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test


def load_imdb(ratio, type_num, return_hg, attack_model):
    # The order of node types: 0 m 1 d 2 a 3 w
    M, A, D = 4661, 5841, 2270
    pos_num = 50
    path = "../data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.load_npz(path + "m_feat.npz").astype("float32")
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_d = sp.load_npz(path + "d_feat.npz").astype("float32")

    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_d = th.FloatTensor(preprocess_features(feat_d))

    def generate_metapaths(ma, md):
        ma, md = ma.T, md.T
        ma_ = sp.coo_matrix((np.ones(ma.shape[0]), (ma[:, 0], ma[:, 1])), shape=(M, A)).toarray()
        md_ = sp.coo_matrix((np.ones(md.shape[0]), (md[:, 0], md[:, 1])), shape=(M, D)).toarray()

        mam = np.matmul(ma_, ma_.T) > 0
        mam = sp.coo_matrix(mam)
        mdm = np.matmul(md_, md_.T) > 0
        mdm = sp.coo_matrix(mdm)
        return mam, mdm

    def generate_pos(mam, mdm):
        mam = mam / mam.sum(axis=-1).reshape(-1, 1)
        mdm = mdm / mdm.sum(axis=-1).reshape(-1, 1)
        all = (mam + mdm).A.astype("float32")

        pos = np.zeros((M, M))
        k = 0
        for i in range(len(all)):
            one = all[i].nonzero()[0]
            if len(one) > pos_num:
                oo = np.argsort(-all[i, one])
                sele = one[oo[:pos_num]]
                pos[i, sele] = 1
                k += 1
            else:
                pos[i, one] = 1
        pos = pos + np.eye(M)
        pos = sp.coo_matrix(pos)
        return pos

    def gen_neibor(edges):
        e_dict = {}
        for i in edges:
            if i[0] not in e_dict:
                e_dict[int(i[0])] = []
                e_dict[int(i[0])].append(int(i[1]))
            else:
                e_dict[int(i[0])].append(int(i[1]))
        e_keys = sorted(e_dict.keys())
        e_nei = [e_dict[i] for i in e_keys]
        e_nei = np.array([np.array(i) for i in e_nei], dtype=int)
        return e_nei

    attack_ratio = 0.1

    if attack_model == 'random':
        ma = np.loadtxt(path + 'ma.txt', dtype=int).T
        md = np.loadtxt(path + 'md.txt', dtype=int).T

        ma = EdgePerturb(ma, attack_ratio, M, A)
        md = EdgePerturb(md, attack_ratio, M, D)

        mam, mdm = generate_metapaths(ma, md)
        pos = generate_pos(mam, mdm)

    elif attack_model == 'CLGA':
        ma = np.loadtxt(path + 'ma_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        mam, mdm = generate_metapaths(ma, md)
        pos = generate_pos(mam, mdm)

    elif attack_model == 'SHAC':
        ma = np.loadtxt(path + 'ma_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        mam, mdm = generate_metapaths(ma, md)
        pos = generate_pos(mam, mdm)

    # mam, mdm = generate_metapaths(ma, md)
    # ma_ = comple_isolated_nodes(ma)
    nei_a = gen_neibor(ma)
    nei_d = gen_neibor(md)

    # pos = generate_pos(mam, mdm)
    # DGL he graph
    hg = dgl.heterograph({
        ('movie', 'ma', 'actor'): (ma[0], ma[1]),
        ('actor', 'am', 'movie'): (ma[1], ma[0]),
        ('movie', 'md', 'direct'): (md[0], md[1]),
        ('direct', 'dm', 'movie'): (md[1], md[0])
    })
    # hg = dgl.to_bidirected(hg)
    hg.nodes['movie'].data['h'] = feat_m
    hg.nodes['actor'].data['h'] = feat_a
    hg.nodes['direct'].data['h'] = feat_d

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]

    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    canonical_etypes = [('movie', 'ma', 'actor'), ('movie', 'md', 'direct')]
    main_type = 'movie'
    if return_hg == True:
        return hg, canonical_etypes, main_type, [feat_m, feat_a, feat_d], [mam, mdm], pos, label, train, val, test
    else:
        return [nei_a, nei_d], [feat_m, feat_a, feat_d], [mam, mdm], pos, label, train, val, test


def load_data_attack(dataset, ratio, type_num, return_hg, attack_model):
    if dataset == "acm":
        data = load_acm(ratio, type_num, return_hg, attack_model)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num, return_hg)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num, return_hg)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num, return_hg, attack_model)
    elif dataset == "imdb":
        data = load_imdb(ratio, type_num, return_hg, attack_model)
    return data
