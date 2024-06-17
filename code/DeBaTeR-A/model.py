import torch
from torch import nn
import torch.nn.functional as F

class TimeEncoder(nn.Module):
    def __init__(self, total_d, tdim, tlen, device):
        super(TimeEncoder, self).__init__()
        self.total_d = total_d
        d = int(total_d / tdim)
        self.d = d
        self.tlen = tlen
        self.tdim = tdim
        self.device = device
        Ws = []
        for i in range(tdim):
            if i == tdim - (total_d % tdim): d += 1
            W = nn.parameter.Parameter(torch.empty((tlen[i], d)))
            nn.init.xavier_normal_(W)
            Ws.append(W)
        self.Ws = nn.ParameterList(Ws)

    def forward(self, T):
        ts = []
        for i in range(self.tdim):
            t = torch.squeeze(T[:, i])
            t = F.one_hot(t, num_classes = self.tlen[i]).float()
            e_t = t @ self.Ws[i]
            ts.append(e_t)
        t_emb = torch.cat(ts, dim = -1)
        return t_emb

    def l2_loss(self, T):
        t_emb = self.forward(T)
        return t_emb.norm(dim=1).mean()


class Model(nn.Module):
    def __init__(self, n, m, d, nlayer, tdim, tlen, device, eps=0.1):
        super(Model, self).__init__()
        self.n = n
        self.m = m
        self.d = d
        self.eps = eps
        self.nlayer = nlayer
        self.device = device
        self.user_embeddings = nn.parameter.Parameter(torch.empty(n, d))
        self.item_embeddings = nn.parameter.Parameter(torch.empty(m, d))
        self.user_embeddings_forward = torch.empty(n, d)
        self.item_embeddings_forward = torch.empty(m, d)
        self.user_embeddings_perturbed = torch.empty(n, d)
        self.item_embeddings_perturbed = torch.empty(m, d)
        self.user_embeddings_perturbed2 = torch.empty(n, d)
        self.item_embeddings_perturbed2 = torch.empty(m, d)
        nn.init.xavier_uniform_(self.user_embeddings)
        nn.init.xavier_uniform_(self.item_embeddings)
        self.timeEncoder = TimeEncoder(d, tdim, tlen, device)
        self.similarity = nn.CosineSimilarity(dim=1)

    def forward(self, u_id, i_id, T, beta):
        t_emb = self.timeEncoder(T)

        u_t = F.normalize((self.user_embeddings[u_id, :]+ t_emb), dim=-1)
        i_t = F.normalize((self.item_embeddings[i_id, :]+ t_emb), dim=-1)
        s = (u_t * i_t).sum(dim=-1)
        s = (s + 1) / 2
        s[s < beta] = 0

        A_r = torch.sparse_coo_tensor(torch.stack([u_id, i_id], dim=0), s).coalesce()
        u_embs = [self.user_embeddings]
        i_embs = [self.item_embeddings]
        for l in range(self.nlayer):
            u_embs.append(torch.sparse.mm(A_r, i_embs[l]))
            i_embs.append(torch.sparse.mm(A_r.t(), u_embs[l]))
        self.user_embeddings_forward = torch.mean(torch.stack(u_embs[1:], dim=0), dim=0)
        self.item_embeddings_forward = torch.mean(torch.stack(i_embs[1:], dim=0), dim=0)

        u_embs = [self.user_embeddings]
        i_embs = [self.item_embeddings]
        for l in range(self.nlayer):
            u_emb = torch.sparse.mm(A_r, i_embs[l])
            i_emb = torch.sparse.mm(A_r.t(), u_embs[l])
            random_noise = F.normalize(u_emb[torch.randperm(u_emb.shape[0]), :], dim=1) * self.eps
            u_emb += random_noise
            random_noise = F.normalize(i_emb[torch.randperm(i_emb.shape[0]), :], dim=1) * self.eps
            i_emb += random_noise
            u_embs.append(u_emb)
            i_embs.append(i_emb)
        self.user_embeddings_perturbed = torch.mean(torch.stack(u_embs[1:], dim=0), dim=0)
        self.item_embeddings_perturbed = torch.mean(torch.stack(i_embs[1:], dim=0), dim=0)

        u_embs = [self.user_embeddings]
        i_embs = [self.item_embeddings]
        for l in range(self.nlayer):
            u_emb = torch.sparse.mm(A_r, i_embs[l])
            i_emb = torch.sparse.mm(A_r.t(), u_embs[l])
            random_noise = F.normalize(u_emb[torch.randperm(u_emb.shape[0]), :], dim=1) * self.eps
            u_emb += random_noise
            random_noise = F.normalize(i_emb[torch.randperm(i_emb.shape[0]), :], dim=1) * self.eps
            i_emb += random_noise
            u_embs.append(u_emb)
            i_embs.append(i_emb)
        self.user_embeddings_perturbed2 = torch.mean(torch.stack(u_embs[1:], dim=0), dim=0)
        self.item_embeddings_perturbed2 = torch.mean(torch.stack(i_embs[1:], dim=0), dim=0)
        return

    def predict(self, u_id, t_u):
        t_emb = self.timeEncoder(t_u)
        s = (self.user_embeddings_forward[u_id]+t_emb).unsqueeze(dim=1) * \
            (self.item_embeddings_forward.unsqueeze(dim=0)+t_emb.unsqueeze(dim=1))
        return s.sum(dim=-1)

    def l2_loss(self, u_id, pos_id, neg_id, t):
        return self.user_embeddings_forward[u_id].norm(dim=1).mean() \
               + self.item_embeddings_forward[pos_id].norm(dim=1).mean() \
               + self.item_embeddings_forward[neg_id].norm(dim=1).mean() \
               + self.timeEncoder.l2_loss(t)
    
    def get_loss(self, u_id, pos_id, neg_id, t):
        t_emb = self.timeEncoder(t)
        u_pos_t = self.user_embeddings_forward[u_id]+t_emb
        i_pos_t = self.item_embeddings_forward[pos_id]+t_emb
        i_neg_t = self.item_embeddings_forward[neg_id]+t_emb
        bpr = self.bpr_loss(u_pos_t, i_pos_t, i_neg_t)
        alignment = self.alignment_loss(u_pos_t, i_pos_t)
        uniformity = (self.uniformity_loss(u_pos_t) + self.uniformity_loss(i_neg_t)) / 2
        return bpr, alignment, uniformity

    def bpr_loss(self, x, y, y_neg):
        s = x.multiply(y).sum(dim=1)-x.multiply(y_neg).sum(dim=1)
        l = -torch.log(F.sigmoid(s)+1e-12).mean()
        return l

    def alignment_loss(self, x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniformity_loss(self, x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def cl_loss(self, u_id, i_id, tau=0.2):
        u_id = u_id.unique()
        i_id = i_id.unique()
        u_emb = self.user_embeddings_perturbed2[u_id]
        u_emb1 = self.user_embeddings_perturbed[u_id]
        i_emb = self.item_embeddings_perturbed2[i_id]
        i_emb1 = self.item_embeddings_perturbed[i_id]
        u_emb = F.normalize(u_emb, dim=1)
        i_emb = F.normalize(i_emb, dim=1)
        u_emb1 = F.normalize(u_emb1, dim=1)
        i_emb1 = F.normalize(i_emb1, dim=1)
        s_u = u_emb.multiply(u_emb1)/tau
        s_i = i_emb.multiply(i_emb1)/tau
        t_u = u_emb.matmul(u_emb1.t())/tau
        t_i = i_emb.matmul(i_emb1.t())/tau
        s_u = torch.exp(s_u)/(torch.exp(t_u).sum(dim=1).unsqueeze(dim=1))
        s_i = torch.exp(s_i)/(torch.exp(t_i).sum(dim=1).unsqueeze(dim=1))
        l = -torch.log(s_u).mean()-torch.log(s_i).mean()
        return l
