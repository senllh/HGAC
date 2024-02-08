import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
import torch

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

def get_inverse_id(index, num):
    return_list = []
    for i in range(num):
        if i not in index:
            return_list.append(i)
    return return_list

def EdgePerturb(edge_index, aug_ratio, src_num, dst_num):
    edge_index = torch.Tensor(edge_index.T)
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
    return edge_index.cpu().numpy()

def EdgePerturb2(edge_index, aug_ratio, src_num, dst_num):
    edge_index = torch.Tensor(edge_index.T)
    _, edge_num = edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    src_unif = torch.ones(1, src_num)
    dst_unif = torch.ones(1, dst_num)

    # # 随机抽样
    unif = torch.ones(edge_num)
    remove_idx = unif.multinomial((permute_num), replacement=False)
    remove_idx2 = unif.multinomial((permute_num), replacement=False)
    # add_src_idx = src_unif.multinomial(permute_num, replacement=True)
    add_src_idx = edge_index[0, remove_idx2.unsqueeze(0)]
    add_dst_idx = dst_unif.multinomial(permute_num, replacement=True)
    add_edge_idx = torch.cat([add_src_idx, add_dst_idx], dim=0)
    keep_id = get_inverse_id(remove_idx.tolist(), edge_num)

    edge_index = torch.cat((edge_index[:, keep_id], add_edge_idx), dim=1)
    return edge_index.cpu().numpy()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(ratio, type_num, return_hg, attack_model, attack_ratio):
    # The order of node types: 0 p 1 a 2 s
    # path = "../../data/acm/"
    path = "../data/acm/"
    P, A, S = 4019, 7167, 60
    pos_num = 5
    # import os
    # print(os.path.exists("../data"))
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    # 加载原始数据

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
        pos = pos + np.eye(P)
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

    if attack_model == 'random':
        pa = np.genfromtxt(path + "pa.txt", dtype=int)
        ps = np.genfromtxt(path + "ps.txt", dtype=int)

        pa = EdgePerturb(pa, attack_ratio, P, A)
        ps = EdgePerturb(ps, attack_ratio, P, S)

        pap, psp = generate_metapaths(pa, ps)
        nei_a = gen_neibor(pa.T)
        nei_s = gen_neibor(ps.T)
        pos = generate_pos(pap, psp)

    elif attack_model == 'CLGA':
        pa = np.loadtxt(path + 'pa_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        ps = np.loadtxt(path + 'ps_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        nei_a = gen_neibor(pa.T)
        nei_s = gen_neibor(ps.T)
        pap, psp = generate_metapaths(pa, ps)
        pos = generate_pos(pap, psp)

    elif attack_model == 'SHAC':
        pa = np.loadtxt(path + 'pa_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        ps = np.loadtxt(path + 'ps_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        nei_a = gen_neibor(pa.T)
        nei_s = gen_neibor(ps.T)
        pap, psp = generate_metapaths(pa, ps)
        pos = generate_pos(pap, psp)

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)

    nei_a = [th.LongTensor(i.astype('int')) for i in nei_a]
    nei_s = [th.LongTensor(i.astype('int')) for i in nei_s]

    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], pos, label, train, val, test

def load_dblp(ratio, type_num, return_hg=False, attack_model='random', attack_ratio=0.1):
    # The order of node types: 0 a 1 p 2 c 3 t
    # The order of node types: 0 a 1 p 2 c 3 t
    A, P, C, T = 4057, 14328, 20, 7723
    pos_num = 1000#1000
    #['apa', 'apcpa', 'aptpa']
    def generate_metapaths(pa, pc, pt):
        pa, pc, pt = pa.T, pc.T, pt.T
        pa_ = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
        pc_ = sp.coo_matrix((np.ones(pc.shape[0]), (pc[:, 0], pc[:, 1])), shape=(P, C)).toarray()
        pt_ = sp.coo_matrix((np.ones(pt.shape[0]), (pt[:, 0], pt[:, 1])), shape=(P, T)).toarray()

        apa = np.matmul(pa_.T, pa_) > 0
        apa = sp.coo_matrix(apa)

        apc = np.matmul(pa_.T, pc_) > 0
        apcpa = np.matmul(apc, apc.T) > 0
        apcpa = sp.coo_matrix(apcpa)

        apt = np.matmul(pa_.T, pt_) > 0
        aptpa = np.matmul(apt, apt.T) > 0
        aptpa = sp.coo_matrix(aptpa)
        return apa, apcpa, aptpa
    
    def generate_pos(apa, apcpa, aptpa):
        apa = apa / apa.sum(axis=-1).reshape(-1, 1)
        apcpa = apcpa / apcpa.sum(axis=-1).reshape(-1, 1)
        aptpa = aptpa / aptpa.sum(axis=-1).reshape(-1, 1)
        all = (apa + apcpa + aptpa).A.astype("float32")

        pos = np.zeros((A, A))
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
        pos = pos + np.eye(A)
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

    path = "../data/dblp/"
    # path = "../../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)

    if attack_model == 'random':
        pa = np.loadtxt(path + 'pa.txt', dtype=int)
        pc = np.loadtxt(path + 'pc.txt', dtype=int)
        pt = np.loadtxt(path + 'pt.txt', dtype=int)

        ap = pa[:, [1, 0]]
        ap = EdgePerturb(ap, attack_ratio, A, P)
        pa = ap[[1, 0]]
        pc = EdgePerturb(pc, attack_ratio, P, C)
        pt = EdgePerturb(pt, attack_ratio, P, T)

        apa, apcpa, aptpa = generate_metapaths(pa, pc, pt)
        nei_p = gen_neibor(ap.T)
        pos = generate_pos(apa, apcpa, aptpa)
    elif attack_model == 'CLGA':
        pa = np.loadtxt(path + 'pa_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        pc = np.loadtxt(path + 'pc_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        pt = np.loadtxt(path + 'pt_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        apa, apcpa, aptpa = generate_metapaths(pa, pc, pt)
        ap = pa[[0, 1 ]]
        nei_p = gen_neibor(ap.T)
        pos = generate_pos(apa, apcpa, aptpa)
    elif attack_model == 'SHAC':
        pa = np.loadtxt(path + 'pa_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        pc = np.loadtxt(path + 'pc_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        pt = np.loadtxt(path + 'pt_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        apa, apcpa, aptpa = generate_metapaths(pa, pc, pt)
        ap = pa[[0, 1 ]]
        nei_p = gen_neibor(ap.T)
        pos = generate_pos(apa, apcpa, aptpa)

    # 构建原始图
    node_num = type_num[0] + type_num[1] \
               + type_num[2] + type_num[3]
    pa[0], pc[0], pt[0] = pa[1] + type_num[0], pc[1] + type_num[0], pt[1] + type_num[0]
    pa[1] = pa[1] + type_num[0] + type_num[1]
    pt[1] = pt[1] + type_num[0] + type_num[1] + type_num[2]
    ori_g = np.concatenate([pa, pc, pt], axis=1)
    ori_g = th.LongTensor(ori_g)
    ori_g = th.sparse_coo_tensor(ori_g, th.ones(ori_g.size(1)), (node_num, node_num))  # .to_dense()
    a_id = (0, type_num[0])
    p_id = (type_num[0], type_num[0] + type_num[1])
    c_id = (type_num[0] + type_num[1], node_num - type_num[3])
    t_id = (node_num - type_num[3], node_num)

    edges_id = [(p_id[0], p_id[1], a_id[0], a_id[1]),
                (p_id[0], p_id[1], c_id[0], c_id[1]),
                (p_id[0], p_id[1], t_id[0], t_id[1])]

    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.eye(type_num[1])
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    nei_p = [th.LongTensor(i) for i in nei_p]
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    if return_hg == True:
        return ori_g, edges_id, [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test
    else:
        return [nei_p], [feat_a, feat_p], [apa, apcpa, aptpa], pos, label, train, val, test


def load_aminer(ratio, type_num, return_hg=False, attack_model='CLGA', attack_ratio=0.1):
    # The order of node types: 0 p 1 a 2 r
    P, A, R = 6564, 13329, 35890
    pos_num = 15
    path = "../data/aminer/"

    def generate_metapaths(pa, pr):
        pa, pr = pa.T, pr.T
        pa_ = sp.coo_matrix((np.ones(pa.shape[0]), (pa[:, 0], pa[:, 1])), shape=(P, A)).toarray()
        pr_ = sp.coo_matrix((np.ones(pr.shape[0]), (pr[:, 0], pr[:, 1])), shape=(P, R)).toarray()

        pap = np.matmul(pa_, pa_.T) > 0
        pap = sp.coo_matrix(pap)
        prp = np.matmul(pr_, pr_.T) > 0
        prp = sp.coo_matrix(prp)
        return pap, prp

    def generate_pos(pap, prp):
        pap = pap / pap.sum(axis=-1).reshape(-1, 1)
        prp = prp / prp.sum(axis=-1).reshape(-1, 1)
        all = (pap + prp).A.astype("float32")

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
        pos = pos + np.eye(P)
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

    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    # nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    # nei_r = np.load(path + "nei_r.npy", allow_pickle=True)

    if attack_model == 'random':
        pa = np.loadtxt(path + 'pa.txt', dtype=int)
        pr = np.loadtxt(path + 'pr.txt', dtype=int)

        pa = EdgePerturb(pa, attack_ratio, P, A)
        pr = EdgePerturb(pr, attack_ratio, P, R)

        # pa = EdgePerturb2(pa, attack_ratio, P, A)
        # pr = EdgePerturb2(pr, attack_ratio, P, R)

        pap, prp = generate_metapaths(pa, pr)
        nei_a = gen_neibor(pa.T)
        nei_r = gen_neibor(pr.T)
        pos = generate_pos(pap, prp)
    elif attack_model == 'CLGA':
        pa = np.loadtxt(path + 'pa_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        pr = np.loadtxt(path + 'pr_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        pap, prp = generate_metapaths(pa, pr)
        nei_a = gen_neibor(pa.T)
        nei_r = gen_neibor(pr.T)
        pos = generate_pos(pap, prp)

    elif attack_model == 'SHAC':
        pa = np.loadtxt(path + 'pa_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        pr = np.loadtxt(path + 'pr_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)

        pap, prp = generate_metapaths(pa, pr)
        nei_a = gen_neibor(pa.T)
        nei_r = gen_neibor(pr.T)
        pos = generate_pos(pap, prp)

    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p = sp.eye(type_num[0])
    feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]

    # 构建原始图
    node_num = type_num[0] + type_num[1] + type_num[2]
    pa[1] = pa[1] + type_num[0]
    pr[1] = pr[1] + type_num[0] + type_num[1]
    ori_g = np.concatenate([pa, pr], axis=1)
    ori_g = th.LongTensor(ori_g)
    ori_g = th.sparse_coo_tensor(ori_g, th.ones(ori_g.size(1)), (node_num, node_num))#.to_dense()
    edges_id = [(0, type_num[0], type_num[0], type_num[0] + type_num[1]),
               (0, type_num[0], type_num[0] + type_num[1], node_num)]

    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    if return_hg == True:
        return ori_g, edges_id, [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test
    else:
        return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], pos, label, train, val, test

def load_freebase(ratio, type_num, return_hg=False, attack_model='CLGA', attack_ratio=0.1):
    # The order of node types: 0 m 1 d 2 a 3 w
    M, D, A, W = 3492, 2502, 33401, 4459
    pos_num = 80
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

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

    if attack_model == 'random':
        ma = np.loadtxt(path + 'ma.txt', dtype=int)
        md = np.loadtxt(path + 'md.txt', dtype=int)
        mw = np.loadtxt(path + 'mw.txt', dtype=int)

        ma = EdgePerturb(ma, attack_ratio, M, A)
        md = EdgePerturb(md, attack_ratio, M, D)
        mw = EdgePerturb(mw, attack_ratio, M, W)

        mam, mdm, mwm = generate_metapaths(ma, md, mw)

        nei_a = gen_neibor(ma.T)
        nei_d = gen_neibor(md.T)
        nei_w = gen_neibor(mw.T)

        pos = generate_pos(mam, mdm, mwm)

    elif attack_model == 'CLGA':
        ma = np.loadtxt(path + 'ma_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mw = np.loadtxt(path + 'mw_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mam, mdm, mwm = generate_metapaths(ma, md, mw)
        nei_a = gen_neibor(ma.T)
        nei_d = gen_neibor(md.T)
        nei_w = gen_neibor(mw.T)
        pos = generate_pos(mam, mdm, mwm)

    elif attack_model == 'SHAC':
        ma = np.loadtxt(path + 'ma_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mw = np.loadtxt(path + 'mw_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mam, mdm, mwm = generate_metapaths(ma, md, mw)
        nei_a = gen_neibor(ma.T)
        nei_d = gen_neibor(md.T)
        nei_w = gen_neibor(mw.T)
        pos = generate_pos(mam, mdm, mwm)

    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_d, nei_a, nei_w], [feat_m, feat_d, feat_a, feat_w], [mdm, mam, mwm], pos, label, train, val, test

def load_imdb(ratio, type_num, attack_model='random', attack_ratio=0.1):
    # The order of node types: 0 m 1 d 2 a 3 w
    M, A, D = 4661, 5841, 2270
    pos_num = 50
    neg_num = 2
    path = "../data/imdb/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_m = sp.load_npz(path + "m_feat.npz").astype("float32")
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_d = sp.load_npz(path + "d_feat.npz").astype("float32")

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
        e_nei = np.array([np.array(i) for i in e_nei], dtype=object)
        return e_nei

    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.

    # ma = np.loadtxt(path + 'ma.txt', dtype=int)
    # md = np.loadtxt(path + 'md.txt', dtype=int)
    #
    # mam, mdm = generate_metapaths(ma, md)
    #
    # nei_a = gen_neibor(ma)
    # nei_d = gen_neibor(md)
    #
    # pos = generate_pos(mam, mdm)
    if attack_model == 'random':
        ma = np.loadtxt(path + 'ma.txt', dtype=int).T
        md = np.loadtxt(path + 'md.txt', dtype=int).T
        ma = EdgePerturb(ma.T, attack_ratio, M, A)
        md = EdgePerturb(md.T, attack_ratio, M, D)
        mam, mdm = generate_metapaths(ma, md)
        nei_a = gen_neibor(ma.T)
        nei_d = gen_neibor(md.T)
        pos = generate_pos(mam, mdm)

    elif attack_model == 'CLGA':
        ma = np.loadtxt(path + 'ma_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_CLGA_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mam, mdm = generate_metapaths(ma, md)
        nei_a = gen_neibor(ma.T)
        nei_d = gen_neibor(md.T)
        pos = generate_pos(mam, mdm)

    elif attack_model == 'SHAC':
        ma = np.loadtxt(path + 'ma_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        md = np.loadtxt(path + 'md_SHAC_gumbel_' + str(attack_ratio) + '.txt', dtype=int)
        mam, mdm = generate_metapaths(ma, md)
        nei_a = gen_neibor(ma.T)
        nei_d = gen_neibor(md.T)
        pos = generate_pos(mam, mdm)

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_d = [th.LongTensor(i.astype(float)) for i in nei_d]
    nei_a = [th.LongTensor(i.astype(float)) for i in nei_a]

    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))

    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))

    pos = sparse_mx_to_torch_sparse_tensor(pos)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_d], [feat_m, feat_a, feat_d], [mam, mdm], pos, label, train, val, test


def load_data_attack(dataset, ratio, type_num, return_hg, attack_model, attack_ratio):
    if dataset == "acm":
        data = load_acm(ratio, type_num, return_hg, attack_model, attack_ratio=attack_ratio)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num, attack_model, attack_ratio=attack_ratio)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num, return_hg, attack_model, attack_ratio=attack_ratio)
    elif dataset == "imdb":
        data = load_imdb(ratio, type_num, attack_model, attack_ratio=attack_ratio)
    return data
