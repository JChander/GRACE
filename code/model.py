from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layer import GraphConvolution, GraphAttentionLayer

class Encoder(nn.Module):
    def __init__(self, nhid, nz) -> None:
        super().__init__()
        self.linear1 = nn.Linear(nhid, nz)
        self.linear2 = nn.Linear(nhid, nz)

    
    def forward(self, x):
        #x = torch.cat((x, y), dim=1)
        mu = F.relu(self.linear1(x))
        logvar = F.relu(self.linear2(x))
        # variance = mu.mean(dim=1)
        # print(logvar.mean(dim=1).sum())
        # print(variance.sum())

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, ns, nhid) -> None:
        super().__init__()
        self.linear_x = nn.Linear(ns, nhid)
    
    def forward(self, s):
        x = F.sigmoid(self.linear_x(s))
        return x


class CausalGNN(nn.Module):
    def __init__(self, nfeat, nhid, nz, ns,dropout, alpha, base_model='gcn', nheads=8, flag=False) -> None:
        super().__init__()
        self.dropout = dropout
        # concat: whether input elu layer
        # encoder 
        self.attention_z = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.encoder = Encoder(nhid, nz)

        # z->s
        self.attention_s = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.linear_s = nn.Linear(nhid + nz, ns)

        # decoder
        self.decoder = Decoder(ns, nfeat)

        # predict
        if base_model == 'gcn':
            self.base_model = GCN(nfeat, ns, nhid, dropout, flag)
        else:
            self.base_model = GAT(nfeat, nhid, dropout, alpha, nheads, ns, flag)
        
        self.nz = nz


    def forward(self, feat, adj, train_ids, stage=None, z_mean=None):
        # x = Wx
        h_z = self.attention_z(feat, adj)
        h_s = self.attention_s(feat, adj)
        # x = F.dropout(h_z, self.dropout, training=self.training)
        x = h_z
        mu = [] 
        logvar = []
        z_sum = 0
        if stage == 'training':
        # encoder 
            mu, logvar = self.encoder(x)
            z = self.reparametrize(mu, logvar)
            z_sum = z.var(dim=1).sum()
            z_mean = torch.mean(z, dim=0).unsqueeze(0)
        else: 
            z = z_mean.repeat(x.size()[0], 1)

        # z->s
        x = h_s
        x = torch.cat((z, x), dim=1)
        s = F.relu(self.linear_s(x))
        # s = F.dropout(s, self.dropout, training=self.training)

        # decoder
        recon_x = self.decoder(s)

        # predict
        output = self.base_model(s, recon_x, adj, train_ids)
        return z_mean, output, recon_x, mu, logvar, z_sum
    
    def reparametrize(self, mu, logvar):
        # # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        # std = 0.5 * torch.exp(logvar)
        # # N(mu, std^2) = N(0, 1) * std + mu
        # z = torch.randn(std.size()) * std + mu
    
        # eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        eps = torch.randn_like(logvar)
        z = mu + eps * torch.exp(logvar/2)
        return z



class GCN(nn.Module):
    def __init__(self, nfeat, ns, nhid, dropout, flag = False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.linear1 = nn.Linear((nhid+ns) * 2, nhid)
        if flag:
            self.linear2 = nn.Linear(nhid, 3)
        else:
            self.linear2 = nn.Linear(nhid, 1)
        #self.dotprodcut = DotProductPredictor(dropout, act = lambda x: x)
        self.flag = flag
        self.dropout = dropout

    def forward(self, s, x, adj, train_ids):
        # tranditional gcn
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, adj))
        x = self.gc2(x, adj)


        # add s
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat((x, s), dim=1)
        x_src = x[train_ids[:, 0]]
        x_dst = x[train_ids[:, 1]]

        x_edges = torch.cat([x_src, x_dst], dim=1)
        output = F.relu(self.linear1(x_edges))
        output = self.linear2(output)
        # x = self.dotprodcut(x)
        if self.flag:
            return output
        else:
            return output.squeeze(1)



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, ns, flag = False):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False) 

        self.linear1 = nn.Linear((nhid * nheads + ns) * 2, nhid * nheads)
        if flag:
            self.linear2 = nn.Linear(nhid * nheads, 3)
        else:
            self.linear2 = nn.Linear(nhid * nheads, 1)
        self.flag = flag

    def forward(self, s, x, adj, train_ids):
        # tranditional GAT
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        x = self.out_att(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)

        # # add s
        x = torch.cat((x, s), dim=1)
        x_src = x[train_ids[:, 0]]
        x_dst = x[train_ids[:, 1]]

        x_edges = torch.cat([x_src, x_dst], dim=1)
        output = F.relu(self.linear1(x_edges))
        if self.flag:
            output = self.linear2(output)
        else:
            output = self.linear2(output).squeeze(1)
        return output


class DotProductPredictor(nn.Module):
	def __init__(self, dropout = 0., act = F.sigmoid) -> None:
		super().__init__()
		self.dropout = dropout
		self.act = act

	def forward(self, h):
		h = F.dropout(h, self.dropout)
		x = torch.mm(h, h.T)
		return self.act(x)

