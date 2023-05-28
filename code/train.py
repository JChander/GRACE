import argparse
from lib2to3.pytree import Base
import torch
import numpy as np
import random
import yaml
import gc
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils import load_sc_data, load_sc_causal_data, accuracy_LP
from model import CausalGNN
import torchmetrics
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

# output to a file

# set seed
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# input params
parser = argparse.ArgumentParser()
with open('param.yaml', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)['gcn']
    for key in config.keys():
        name = '--' + key
        parser.add_argument(name, type=type(config[key]), default=config[key])
args = parser.parse_args()

# use cuda
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# set seed
set_rng_seed(args.seed)

# load data
data_path = "./example/mESC/ExpressionData.csv"
label_path = "./example/mESC/refNetwork.csv"

if args.flag:
    adj_train, feature, train_ids, val_ids, test_ids, train_labels, val_labels, test_labels = load_sc_causal_data(data_path, label_path)
else:
    adj_train, feature, train_ids, val_ids, test_ids, train_labels, val_labels, test_labels = load_sc_data(data_path, label_path)

adj_train = F.normalize(adj_train, p=1, dim=1)

# model
model =CausalGNN(nfeat=feature.shape[1],
                nhid=args.hidden,
                dropout=args.dropout,
                nz=args.nz,
                ns=args.ns,
                alpha=args.alpha, 
                flag=args.flag).to(device)

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
    weight_CE = torch.FloatTensor([1, 23, 23]).to(device)

if args.flag:
    Eval_acc = torchmetrics.Accuracy(task='multiclass', num_classes = 3).to(device)
    Eval_auc = torchmetrics.AUROC(task='multiclass', num_classes = 3).to(device)
    Eval_ap = torchmetrics.AveragePrecision(task="multiclass", num_classes = 3).to(device)

def loss_functiona(output, labels, logvar, mu, feature, recon_x, z_sum, flag):
    if flag:
        pred_loss = F.cross_entropy(output, labels, weight = weight_CE)
    else:
        pred_loss = BCEWL_loss(output, labels)

    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)

    reconstruction_loss = BCE_loss(recon_x, feature)

    loss = pred_loss + KL_divergence * args.KL_loss + reconstruction_loss * args.BCE_loss + args.z_loss * z_sum
    
    return loss


def train_encoder(epoch, model, flag):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    z_mean, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, train_ids, stage='training')
    
    loss_train = loss_functiona(output, train_labels, logvar, mu, feature, recon_x, z_sum, flag)
    if flag:
        acc_train = Eval_acc(output, train_labels)
    else:
        acc_train = accuracy_LP(output, train_labels)
    # loss_train = F.nll_loss(output, label[idx_train])
    
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return z_mean


def train_base_model(epoch, z_mean, flag):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    z_mean, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, train_ids, stage='train_base_model', z_mean=z_mean)
    if flag:
        loss_train = F.cross_entropy(output, train_labels, weight = weight_CE)
        acc_train = Eval_acc(output, train_labels)
    else:
        loss_train = BCEWL_loss(output, train_labels)
        acc_train = accuracy_LP(output, train_labels)
    loss_train.backward(retain_graph=True)
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, z_mean, flag):
    model.eval()
    ##For validationi
    z, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, val_ids, z_mean=z_mean)
    if flag:
        loss_val = F.cross_entropy(output, val_labels, weight = weight_CE)
        acc_val = Eval_acc(output, val_labels)
    else:
        loss_val = BCEWL_loss(output, val_labels)
        acc_val = accuracy_LP(output, val_labels)

    ##For test
    z, output, recon_x, mu, logvar, z_sum = model(feature, adj_train, test_ids, z_mean=z_mean)
    if flag:
        loss_test = F.cross_entropy(output, test_labels, weight = weight_CE)
        acc_test = Eval_acc(output, test_labels)
        roc = Eval_auc(output, test_labels)
        ap = Eval_ap(output, test_labels)
        roc_skl = roc_auc_score(y_true=np.eye(3, dtype = 'uint8')[test_labels.cpu().numpy()], y_score=output.cpu().detach().numpy(), average = 'macro')
    else:
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
    return loss_val, loss_test, acc_val, acc_test, roc, ap, roc_skl, output.cpu().detach().numpy()


z_mean = []
for epoch in range(args.epoches):
    z_mean = train_encoder(epoch, model, args.flag)
z_mean = z_mean.detach()
z_mean.requires_grad = False

min_loss = 1000000
best_acc_val = 0
best_acc_tst = 0
best_roc_test = 0
best_ap_test = 0
best_pred = np.zeros(test_labels.shape)
for epoch in range(args.epoch_base_model):
    train_base_model(epoch, z_mean, args.flag)
    
# test
    if(epoch % 10 == 0):
        loss_val, loss_test, acc_val, acc_test, roc, ap, roc_skl, prediction = test(model, z_mean, args.flag)
        if best_acc_val < acc_val:
            min_loss = loss_val
            best_acc_val = acc_val
            best_acc_tst = acc_test
            best_roc_test = roc
            best_ap_test = ap
            best_pred = prediction

print("best_acc_val:{:.4f}".format(best_acc_val))
print("best_acc_test:{:.4f}".format(best_acc_tst))
print("best_AUC_test:{:.4f}".format(best_roc_test))
print("best_AUPRC_test:{:.4f}".format(best_ap_test))

    
