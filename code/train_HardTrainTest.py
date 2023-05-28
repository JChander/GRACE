import argparse
from lib2to3.pytree import Base
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import yaml
import gc
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils import load_sc_data, normalize_features, sparse_mx_to_torch_sparse_tensor, accuracy_LP
from model import CausalGNN
import torch.optim as optim
import torch.nn.functional as F

# Base model setting
BM = 'gcn'

# set seed
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def read_HardSplit_data(data_path):
    data = pd.read_csv(data_path, header = 0, index_col = 0)
    src = data['TF'].values
    dst = data['Target'].values
    labels = data['Label'].values
    edge_ids = np.stack((src, dst), axis = 1)
    labels = torch.FloatTensor(labels)
    edge_ids = torch.LongTensor(edge_ids)
    return edge_ids, labels

# input params
parser = argparse.ArgumentParser()
with open('param.yaml', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)[BM]
    for key in config.keys():
        name = '--' + key
        parser.add_argument(name, type=type(config[key]), default=config[key])
args = parser.parse_args()

# use cuda
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

datasets = ['mESC']
for dataset in datasets:
    # set seed
    for seed in args.seed:
        set_rng_seed(seed)

        # load data
        test_data = "./example/" + dataset + "/Test_set.csv"
        train_data = "./example/" + dataset + "/Train_set.csv"
        val_data = "./example/" + dataset + "/Validation_set.csv"

        train_ids, train_labels = read_HardSplit_data(train_data)
        val_ids, val_labels = read_HardSplit_data(val_data)
        test_ids, test_labels = read_HardSplit_data(test_data)

        exp_data = pd.read_csv("./example/" + dataset + "/ExpressionData.csv", header = 0, index_col = 0).T
        exp_data = exp_data.transform(lambda x: np.log(x + 1))
        features = exp_data.T.values

        #features = normalize_features(features)
        feature = torch.FloatTensor(features)

        train_pos_len = np.int32(train_labels.numpy().sum())
        train_pos_u = train_ids[:train_pos_len, 0]
        train_pos_v = train_ids[:train_pos_len, 1]
        adj_train  = sp.coo_matrix((np.ones(train_pos_u.shape), (train_pos_u, train_pos_v)), shape = (features.shape[0], features.shape[0]), dtype = np.float32)
        adj_train = adj_train + adj_train.T.multiply(adj_train.T > adj_train) - adj_train.multiply(adj_train.T > adj_train)
        # adj_train = adj_train + sp.eye(adj_train.shape[0])
        adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)

        adj_train = F.normalize(adj_train, p=1, dim=1)

        # model
        model =CausalGNN(nfeat=feature.shape[1],
                        nhid=args.hidden,
                        dropout=args.dropout,
                        nz=args.nz,
                        ns=args.ns,
                        base_model = BM,
                        alpha=args.alpha).to(device)

        optimizer = optim.Adam(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)

        BCE_loss = torch.nn.BCELoss(reduction='mean')
        BCEWL_loss = torch.nn.BCEWithLogitsLoss(reduction = 'mean')

        if args.cuda:
            adj_train = adj_train.to(device)
            feature = feature.to(device)
            train_ids = train_ids.to(device)
            val_ids = val_ids.to(device)
            test_ids = test_ids.to(device)
            train_labels = train_labels.to(device)
            val_labels = val_labels.to(device)
            test_labels = test_labels.to(device)


        def loss_functiona(output, labels, logvar, mu, feature, recon_x, z_sum):
            pred_loss = BCEWL_loss(output, labels)

            KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)

            reconstruction_loss = BCE_loss(recon_x, feature)
            #reconstruction_loss = torch.mean((feature - recon_x).pow(2))    ##For reconstruction loss based on MSE

            loss = pred_loss + KL_divergence * args.KL_loss + reconstruction_loss * args.BCE_loss + args.z_loss * z_sum
            
            return loss


        def train_encoder(epoch, model):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            z_mean, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, train_ids, stage='training')
            loss_train = loss_functiona(output, train_labels, logvar, mu, feature, recon_x, z_sum)
            acc_train = accuracy_LP(output, train_labels)
            loss_train.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(time.time() - t))
            return z_mean


        def train_base_model(epoch, z_mean):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            z_mean, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, train_ids, stage='train_base_model', z_mean=z_mean)
            loss_train = BCEWL_loss(output, train_labels)
            acc_train = accuracy_LP(output, train_labels)
            loss_train.backward(retain_graph=True)
            optimizer.step()

            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(time.time() - t))


        def test(model, z_mean):
            model.eval()
            ##For validationi
            z, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, val_ids, z_mean=z_mean)
            loss_val = BCEWL_loss(output, val_labels)
            acc_val = accuracy_LP(output, val_labels)

            ##For test
            z, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, test_ids, z_mean=z_mean)
            loss_test = BCEWL_loss(output, test_labels)
            acc_test = accuracy_LP(output, test_labels)
            roc = roc_auc_score(test_labels.cpu().numpy(), output.cpu().detach().numpy())
            ap = average_precision_score(test_labels.cpu().numpy(), output.cpu().detach().numpy())

            print("loss_val= {:.4f}".format(loss_val.item()),
                  "acc_val= {:.4f}".format(acc_val.item()),
                  "loss_test= {:.4f}".format(loss_test.item()),
                  "acc_test= {:.4f}".format(acc_test.item()),
                  "AUC_test= {:.4f}".format(roc.item()),
                  "AUPRC_test= {:.4f}".format(ap.item()))
            return loss_val, loss_test, acc_val, acc_test, roc, ap


        z_mean = []
        for epoch in range(args.epoches):
            z_mean = train_encoder(epoch, model)
        z_mean = z_mean.detach()
        z_mean.requires_grad = False

        min_loss = 1000000
        best_acc_val = 0
        best_acc_tst = 0
        best_roc_test = 0
        best_ap_test = 0
        for epoch in range(args.epoch_base_model):
            train_base_model(epoch, z_mean)
            
        # test
            if(epoch % 10 == 0):
                loss_val, loss_test, acc_val, acc_test, roc, ap = test(model, z_mean)

                if best_acc_val < acc_val:
                    min_loss = loss_val
                    best_acc_val = acc_val
                    best_acc_tst = acc_test
                    best_roc_test = roc
                    best_ap_test = ap

        print("best_acc_val:{:.4f}".format(best_acc_val))
        print("best_acc_tst:{:.4f}".format(best_acc_tst))
        print("best_AUC_tst:{:.4f}".format(best_roc_test))
        print("best_AUPRC_tst:{:.4f}".format(best_ap_test))

