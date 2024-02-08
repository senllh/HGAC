import torch
import torch.nn as nn
from GCL.models import DualBranchContrast
import GCL.losses as L
import numpy as np

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        if pos.is_sparse == True:
            dense_pos = pos.to_dense()
        else:
            dense_pos = pos
        # dense_pos = torch.zeros_like(dense_pos)

        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(dense_pos).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(dense_pos).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc


class Local_Global_Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Local_Global_Contrast, self).__init__()
        self.local_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.tau = tau
        self.lam = lam
        for model in self.local_proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        for model in self.global_proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(0.2), mode='G2L').cuda()
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    # def forward(self, z1, z2, c1, c2):
    #
    #     z_proj_1 = self.local_proj(z1)
    #     z_proj_2 = self.local_proj(z2)
    #
    #     c_proj_1 = self.global_proj(c1)
    #     c_proj_2 = self.global_proj(c2)
    #
    #     c_proj_1 = c_proj_1.expand_as(z1)
    #     c_proj_2 = c_proj_2.expand_as(z2)
    #
    #     matrix_12 = self.sim(z_proj_1, c_proj_2)
    #     matrix_21 = self.sim(z_proj_2, c_proj_1)
    #
    #     matrix_12_ = matrix_12.T
    #     matrix_21_ = matrix_21.T
    #
    #     dense_pos = torch.eye(z1.size(0)).cuda()
    #
    #     matrix_12_score = matrix_12 / (torch.sum(matrix_12, dim=1).view(-1, 1) + 1e-8)
    #     lori_12 = -torch.log(matrix_12_score.mul(dense_pos).sum(dim=-1)).mean()
    #
    #     matrix_12_score_ = matrix_12_ / (torch.sum(matrix_12_, dim=1).view(-1, 1) + 1e-8)
    #     lori_12_ = -torch.log(matrix_12_score_.mul(dense_pos).sum(dim=-1)).mean()
    #
    #     matrix_21_score = matrix_21 / (torch.sum(matrix_21, dim=1).view(-1, 1) + 1e-8)
    #     lori_21 = -torch.log(matrix_21_score.mul(dense_pos).sum(dim=-1)).mean()
    #
    #     matrix_12_score_ = matrix_21_ / (torch.sum(matrix_21_, dim=1).view(-1, 1) + 1e-8)
    #     lori_21_ = -torch.log(matrix_12_score_.mul(dense_pos).sum(dim=-1)).mean()
    #
    #     loss_12 = lori_12 * 0.5 + lori_12_ * 0.5
    #     loss_21 = lori_21 * 0.5 + lori_21_ * 0.5
    #
    #     return 0.5 * loss_12 + 0.5 * loss_21

    def forward(self, z1, z2):

        c1, c2 = z1.mean(dim=0), z2.mean(dim=0)
        c1, c2 = c1.unsqueeze(0), c2.unsqueeze(0)

        z_proj_1 = self.local_proj(z1)
        z_proj_2 = self.local_proj(z2)

        idx1 = np.random.permutation(z1.size(0))
        idx2 = np.random.permutation(z1.size(0))
        h3 = z_proj_1[idx1]
        h4 = z_proj_2[idx2]

        c_proj_1 = self.global_proj(c1)
        c_proj_2 = self.global_proj(c2)

        return self.contrast_model(h1=z_proj_1, h2=z_proj_2, g1=c_proj_1, g2=c_proj_2, h3=h3, h4=h4)

class IntraContrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(IntraContrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z, pos):
        dense_pos = pos.to_dense()

        z_proj = self.proj(z)
        matrix_sim = self.sim(z_proj, z_proj)

        matrix_sim = matrix_sim / (torch.sum(matrix_sim, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_sim.mul(dense_pos).sum(dim=-1)).mean()
        return lori_mp