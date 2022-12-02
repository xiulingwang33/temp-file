import sys
sys.path.append('/home/CAMPUS/xwang193/PyGCL-main')
sys.path.append('/home/CAMPUS/xwang193/PyGCL-main/utils')
sys.path.append('/home/CAMPUS/xwang193/PyGCL-main/GCL2')
#
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam

# from GCL.eval import get_split, LREvaluator_mia

# from GCL2.eval import get_split
# from GCL2.eval.logistic_regression import LREvaluator

from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import networkx as nx
# from . import train_test_split
# from . import eval
#
# from . import logistic_regression

from GCL2 import preprocessing

import torch
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from torch.optim import Adam
from sklearn.metrics import f1_score,accuracy_score

from GCL.eval import BaseEvaluator

import pandas as pd

import random
import os
import time



class MLP_mia(nn.Module):
    def __init__(self, emb_matrix,hidden_layer, lay_1_dim,dropout):
        super(MLP_mia, self).__init__()
        """
        hidden_layer: dimension of each hidden layer (list type);
        dropout: dropout rate between fully connected layers.
        """
        self.dropout = dropout

        self.emb_matrix = emb_matrix

        MLP_modules = []
        self.num_layers = len(hidden_layer)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                # MLP_modules.append(nn.Linear(np.shape(emb_matrix)[1], hidden_layer[0]))
                MLP_modules.append(nn.Linear( 2, hidden_layer[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer[-1], 1)

        self.weight_vec = torch.empty(1,np.shape(emb_matrix)[1])

        self.weight_vec=nn.init.normal_(self.weight_vec)

        # print('Z******',self.weight_vec)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')


        # Kaiming/Xavier initialization can not deal with non-zero bias terms
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, edge,emb_matrix0,device):
        print(edge.size())
        emb1= emb_matrix0[edge[:,0],:]
        emb2= emb_matrix0[edge[:,1],:]
        # print(emb1)
        # print('####',emb1.size(), self.weight_vec.size())
        print((self.weight_vec.to(device) * emb1).size(),(self.weight_vec.to(device) * emb2))
        embed_sim1 = torch.cosine_similarity((self.weight_vec.to(device)*emb1),(self.weight_vec.to(device)*emb2),dim=1)
        embed_sim2 = torch.mul((self.weight_vec.to(device) * emb1),(self.weight_vec.to(device) * emb2))
        print(embed_sim1.size(), embed_sim2.size())
        embed_sim2 = torch.sum(embed_sim2,dim=1)
        # print('****',embed_sim1, embed_sim2)
        print('@@@@@',embed_sim1.size(), embed_sim2.size())
        print('%%%%', embed_sim1.t().size(), embed_sim2.t().size())
        interaction = torch.cat((embed_sim1.reshape(-1,1), embed_sim2.reshape(-1,1)), 1)
        # interaction = [embed_sim1.t(), embed_sim2.t()]
        print(interaction.size())
        # interaction=torch.FloatTensor(interaction).to(device)
        output = self.MLP_layers(interaction)

        prediction = self.predict_layer(output)
        return prediction.view(-1)







class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator_mia(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    test_acc = accuracy_score(y_test, y_pred)

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro, 'Acc': test_acc})
                    pbar.update(self.test_interval)

        output_train = classifier(x).detach().cpu().numpy()

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'acc': test_acc,
            'output_train': output_train
        }



def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }



class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index0, edge_index,edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        # print(x, x1, x2)
        # print(edge_index, edge_index1, edge_index2)
        #
        # print(x.size(), x1.size(), x2.size())
        # print(edge_index.size(), edge_index1.size(), edge_index2.size())
        #
        # exit()

        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2,edge_index1,edge_index2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data,edges_train_index, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2,edge_index1,edge_index2 = encoder_model(data.x,data.edge_index,edges_train_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    # print('KKKKK',h1,h1.size(),h1.type())
    # exit()
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item(),edge_index1,edge_index2


def test(encoder_model, data,edges_train_index):
    encoder_model.eval()
    z, z1, z2,_,_ = encoder_model(data.x,data.edge_index,edges_train_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result= LREvaluator_mia()(z, data.y, split)

    print(result)


    return result,z, z1, z2


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device(f'cuda:{os.environ["CUDA_VISIBLE_DEVICES"]}')
    torch.cuda.set_device(1)
    # device = torch.device('cuda')
    path = osp.join(osp.expanduser('~'), 'datasets')
    dt='Cora'
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.2)])

    # print(data.edge_index)
    # print(data.edge_index.type())
    # exit()

    edge_index0_all_oneside = []
    edge_index0_all = []

    edge_index0 = data.edge_index.detach().cpu().numpy()
    edge_index0 = edge_index0.transpose()
    for ed in edge_index0:
        if ed[0] > ed[1]:
            edge_index0_all.append([ed[0], ed[1]])
            continue
        else:
            edge_index0_all.append([ed[0], ed[1]])
            edge_index0_all_oneside.append([ed[0], ed[1]])
    edge_index0_all_oneside = np.array(edge_index0_all_oneside)
    edge_index0_all = np.array(edge_index0_all)

    g = nx.Graph()
    g.add_edges_from(edge_index0_all)
    adj_sparse = nx.to_scipy_sparse_matrix(g)
    random.seed(42)
    train_test_split = preprocessing.mask_test_edges(adj_sparse, test_frac=.3, val_frac=0)
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split  # Unpack train-test split
    # print(adj_train)
    g_train0 = nx.from_scipy_sparse_matrix(
        adj_train)  # new graph object with only non-hidden edges, keep all the original nodes

    edge_tuples0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in g_train0.edges()]
    # print(edge_tuples0)

    train_edges0 = set(edge_tuples0)  # initialize train_edges to have all edges
    train_edges0 = np.array([list(edge_tuple) for edge_tuple in train_edges0])
    # print(train_edges1)


    edge_tuples_test0 = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in test_edges]

    edges_test0 = set(edge_tuples_test0)  # initialize test_edges to have all edges
    edges_test0 = np.array([list(edge_tuple) for edge_tuple in edges_test0])

    res_dir = '%s-grace-mia-mi-weighted-batch' % (dt)

    out = open('%s/%s-edges-train.txt' % (res_dir, dt), 'w')
    for item in train_edges0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('%s/%s-edges-test.txt' % (res_dir, dt), 'w')
    for item in edges_test0:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    # adj = adj_train
    #
    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    train_edges_1=np.concatenate((train_edges0[:,1].reshape(-1,1),train_edges0[:,0].reshape(-1,1)),axis=1)
    train_edges_1=np.transpose(np.array(train_edges_1))
    train_edges_2 = np.transpose(np.array(train_edges0))
    # loop_nodes=np.arange(0,g.number_of_nodes())
    # train_edges_3=np.concatenate((loop_nodes.reshape(-1,1),loop_nodes.reshape(-1,1)),axis=1)
    # train_edges_3 = np.transpose(np.array(train_edges_3))

    edges_train_index=np.concatenate((train_edges_1,train_edges_2),axis=1)


    edges_train_index = torch.from_numpy(np.array(edges_train_index)).long().to(device)





    # exit()

    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    best_valid_loss = 99999999

    with tqdm(total=10, desc='(T)') as pbar:
        for epoch in range(1, 10):
            loss,edge_index1,edge_index2 = train(encoder_model, contrast_model, data,edges_train_index, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

            patience = 100

            if loss < best_valid_loss:
                best_valid_loss = loss
                trail_count = 0
                best_epoch = epoch
                torch.save(encoder_model.state_dict(), os.path.join('./checkpoint',
                                                                    'tmp',
                                                                    f'grace_{dt}_best.pt'))

            else:
                trail_count += 1
                if trail_count > patience:
                    print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break

        encoder_model.load_state_dict(torch.load(os.path.join('./checkpoint', 'tmp',
                                                              f'grace_{dt}_best.pt')))

    test_result,z, z1, z2 = test(encoder_model, data,edges_train_index)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f},Acc={test_result["acc"]:.4f}')

    output_train=test_result["output_train"]


    # res_dir = '%s-grace-mia' % (dt)

    edge_index1 = edge_index1.detach().cpu().numpy()
    edge_index2 = edge_index2.detach().cpu().numpy()

    # print(edge_index0, edge_index1, edge_index2, edges_test)

    print(np.shape(edge_index0), np.shape(edge_index1), np.shape(edge_index2))

    edges1 = []
    edges2 = []

    edges1_idx = []
    edges2_idx = []

    for i in range(np.shape(edge_index1)[1]):
        edges1.append([edge_index1[0][i], edge_index1[1][i]])
        edges1_idx.append(edge_index1[0][i] * np.shape(data.x)[0] + edge_index1[1][i])

    for i in range(np.shape(edge_index2)[1]):
        edges2.append([edge_index2[0][i], edge_index2[1][i]])
        edges2_idx.append(edge_index2[0][i] * np.shape(data.x)[0] + edge_index2[1][i])

    # graph1=nx.Graph()
    # graph1.add_edges_from(edges1)
    # graph2 = nx.Graph()
    # graph2.add_edges_from(edges2)
    #
    # print(nx.number_connected_components(graph1))
    # print(nx.number_connected_components(graph2))
    #
    # exit()



    # edges_train_inter_idx = np.intersect1d(edges0_idx, edges1_idx)
    # edges_train_all_idx = np.union1d(edges1_idx, edges2_idx)
    #
    # # edges_test_idx = np.setdiff1d(edges0_idx, edges_train_all_idx)
    #
    # print('***', edges_train_inter_idx)
    # print(len(edges_train_inter_idx), len(edges_train_all_idx), len(edges_test_idx))
    #
    # edges_train_all = []
    # for i in edges_train_all_idx:
    #     edges_train_all.append([int(i / np.shape(data.x)[0]), int(i % np.shape(data.x)[0])])
    #
    # edges_train_inter = []
    # for i in edges_train_inter_idx:
    #     edges_train_inter.append([int(i / np.shape(data.x)[0]), int(i % np.shape(data.x)[0])])

    edges_train_all=train_edges0

    emb_matrix0 = z.detach().cpu().numpy()
    emb_matrix1=z1.detach().cpu().numpy()
    emb_matrix2 = z2.detach().cpu().numpy()

    with open('./%s/%s-embed0.txt' % (res_dir, dt), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix0)[0], np.shape(emb_matrix0)[1]))
        for item in emb_matrix0:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/%s-embed1.txt' % (res_dir, dt), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix1)[0], np.shape(emb_matrix1)[1]))
        for item in emb_matrix1:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    with open('./%s/%s-embed2.txt' % (res_dir, dt), 'w') as f:
        f.write('%d %d\n' % (np.shape(emb_matrix2)[0], np.shape(emb_matrix1)[1]))
        for item in emb_matrix2:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()


    with open('./%s/%s-output_train.txt' % (res_dir, dt), 'w') as f:
        for item in output_train:
            for jtem in item:
                f.write(str(jtem) + '\t')
            f.write('\n')
        f.close()

    return emb_matrix0,output_train,edges_train_all,edges_test0,data.y,res_dir,device


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    # print(h1,h1.type())
    # h1 = F.normalize(h1)
    # h2 = F.normalize(h2)
    # h1=torch.add_(h1,0.000000001)
    return h1 @ h2.t()


def get_edge_embeddings(edge_list,emb_matrix):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        emb1 = emb_matrix[node1]
        #print(np.shape(emb1))
        emb2 = emb_matrix[node2]
        edge_emb = np.multiply(emb1, emb2)
        sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        sim2 = np.dot(emb1, emb2)

        sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))

        #edge_emb = np.array(emb1) + np.array(emb2)
        # print(np.shape(edge_emb))
        embs.append([sim1,sim2,sim3])
    embs = np.array(embs)
    return embs

def get_edge_posts(edge_list,train_preds):
    embs = []
    for edge in edge_list:
        node1 = edge[0]
        node2 = edge[1]
        pre1 = train_preds[node1]
        #print(np.shape(emb1))
        pre2 = train_preds[node2]

        pre_idx1 = np.argmax(pre1)
        pre_idx2 = np.argmax(pre2)
        train_pres_temp1 = np.sort(pre1)
        train_pres_temp2 = np.sort(pre2)
        if pre_idx1 == label[node1]:
            corr = 1
        else:
            corr = 0

        train_pres1_=([train_pres_temp1[-1], train_pres_temp1[-2], corr])

        if pre_idx2 == label[node2]:
            corr = 1
        else:
            corr = 0
        train_pres2_ = ([train_pres_temp2[-1], train_pres_temp2[-2], corr])

        edge_emb = np.multiply(train_pres1_, train_pres2_)
        #edge_emb = np.array(emb1) + np.array(emb2)
        print(np.shape(edge_emb))

        emb1=train_pres1_
        emb2 = train_pres2_

        sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        sim2 = np.dot(emb1, emb2)

        sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))


        embs.append([sim1,sim2,sim3])
    embs = np.array(embs)

    return embs







if __name__ == '__main__':


    emb_matrix, output_train,edges_train_all,edges_test_all,label,res_dir,device=main()

    # print(emb_matrix)

    # exit()

    # edges_train_inter=np.array(edges_train_inter)
    edges_train_all=np.array(edges_train_all)
    edges_test_all=np.array(edges_test_all)

    train_preds=output_train

    train_range1 = list(np.arange(np.shape(edges_train_all)[0]))
    # train_range2 = list(np.arange(np.shape(edges_train_inter)[0]))

    # Train-set edge embeddings
    train_preds_sampled_idx1 = np.array(random.sample(train_range1, np.shape(edges_test_all)[0]))
    # train_preds_sampled_idx2 = np.array(random.sample(train_range2, np.shape(edges_test_all)[0]))

    print(train_preds_sampled_idx1)

    # train_preds_sampled1 = np.array(edges_train_all)[train_preds_sampled_idx1]
    train_edges_sampled1 = np.array(edges_train_all)[train_preds_sampled_idx1,:]

    # train_preds_sampled2 = np.array(edges_train_all)[train_preds_sampled_idx2]
    # train_edges_sampled2 = np.array(edges_train_inter)[train_preds_sampled_idx2,:]

    print(train_edges_sampled1)
    print(edges_test_all)



    ylabel = [1] * len(train_preds_sampled_idx1) + [0] * len(train_preds_sampled_idx1)

    from sklearn.model_selection import train_test_split

    train_edges_list = train_edges_sampled1
    test_edges_list = np.array(edges_test_all)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    print(np.shape(y_label_test))



    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(train_edges_sampled1, y_label_train,
                                                                                test_size=0.3, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(edges_test_all, y_label_test,
                                                                            test_size=0.3, random_state=42)

    train_edges = torch.LongTensor(train_edges_sampled1).to(device)
    test_edges = torch.LongTensor(edges_test_all).to(device)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

    lay_1_dim = np.shape(X_train)[0]

    train_idx = np.shape(X_train)[0]
    train_idx_suffle=random.sample(range(train_idx),train_idx)
    test_idx =np.shape(y_test)[0]

    X_train = torch.LongTensor(X_train).to(device)
    X_test = torch.LongTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)



    hidden_layer_MLP=[128, 64, 32]
    dropout=0

    emb_matrix=torch.FloatTensor(emb_matrix).to(device)

    MLP_model = preprocessing.MLP_mia(emb_matrix,hidden_layer_MLP, lay_1_dim,dropout)
    MLP_model.to(device)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(MLP_model.parameters(), lr=0.001)



    accs=[]
    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    for epoch in range(1000):
        MLP_model.train()  # Enable dropout (if have).
        start_time = time.time()

        egs = X_train[train_idx_suffle]

        labels = y_train[train_idx_suffle][:,2].float().to(device)

        MLP_model.zero_grad()
        prediction = MLP_model(egs, emb_matrix, device)

        print(prediction.size(), labels.size())
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        print(loss)


        acc=0

        MLP_model.eval()
        eg = X_test
        label = y_test[:,2].cpu().detach().numpy()
        prediction = MLP_model(eg,emb_matrix,device).cpu().detach().numpy()

        for j in range(len(label)):
            if prediction[j]>=0.5:
                if label[j]==1:
                    acc+=1
            else:
                if label[j] == 0:
                    acc += 1

        accs.append(acc/test_idx)

    print(max(accs))

