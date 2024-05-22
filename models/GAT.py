#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8,dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True ,device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GATConv(nfeat,nhid,heads,dropout=dropout)
        self.gc2 = GATConv(heads*nhid, nclass, concat=False, dropout=dropout)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def forward(self, x, edge_index, edge_weight=None): 
        x = F.dropout(x, p=self.dropout, training=self.training)    # optional
        # x = F.elu(self.gc1(x, edge_index, edge_weight))   # may apply later 
        x = F.elu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index, edge_weight)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x,dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False,  finetune=False, attach=None):
        if initialize:
            self.initialize()
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)


        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            if finetune==True:
                    self.fientune(self.labels, idx_train, idx_val, attach, train_iters, verbose)
            else:
                    self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
    
    def fientune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose):
        # idx1 = idx_train[:-len(idx_attach)]
        # idx2 = idx_train[-len(idx_attach):]
        # idx1 = [item for item in idx_train if item not in idx_attach]

        idx_train_set = set(idx_train)
        idx_attach_set = set(idx_attach)
        idx1 = list(idx_train_set - idx_attach_set)
        idx2 = idx_attach

        idx1 = torch.tensor(idx1).to(self.device)
        idx2 = torch.tensor(idx2).to(self.device)
        

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx1], labels[idx1])
            probs = F.softmax(output[idx2], dim=1)
            target_probs = probs[range(len(labels[idx2])), labels[idx2]]
            loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

            # Combining the normal and adversarial losses
            loss_train = loss_train + loss_train_2
            loss_train.backward()
            optimizer.step()
            # if i % 10 == 0:
            #     print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
    
    # def finetune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose):
        # Prepare index sets for training
        # idx_train_set = set(idx_train)
        # idx_attach_set = set(idx_attach)
        # idx1 = list(idx_train_set - idx_attach_set)
        # idx2 = list(idx_attach_set)

        # # Extract subgraph for the combined indices
        # idx_subgraph = list(set(idx1 + idx2))
        # subgraph_edge_index, subgraph_edge_weight = subgraph(
        #     subset=idx_subgraph,
        #     edge_index=self.edge_index,
        #     edge_attr=self.edge_weight,
        #     relabel_nodes=True,
        #     num_nodes=self.features.size(0),
        # )

        # # Convert lists to tensor and map to device
        # idx1 = torch.tensor([idx_subgraph.index(x) for x in idx1]).to(self.device)
        # idx2 = torch.tensor([idx_subgraph.index(x) for x in idx2]).to(self.device)

        # # Create optimizer
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # # Training iterations
        # for i in range(train_iters):
        #     self.train()
        #     optimizer.zero_grad()

        #     # Forward pass on the subgraph
        #     output, x = self.forward(
        #         self.features[idx_subgraph].to(self.device),
        #         subgraph_edge_index,
        #         subgraph_edge_weight
        #     )

        #     # Calculate loss on idx1 (training nodes not in idx_attach)
        #     loss_train = F.nll_loss(output[idx1], labels[idx1])

        #     # Calculate the probability loss for idx2 (training nodes in idx_attach)
        #     probs = F.softmax(output[idx2], dim=1)
        #     target_probs = probs[torch.arange(len(labels[idx2])), labels[idx2]]
        #     loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

        #     # Combine losses and backpropagate
        #     loss_train = loss_train + loss_train_2
        #     loss_train.backward()
        #     optimizer.step()

        #     # if verbose and i % 10 == 0:
        #     if i % 10 == 0:
        #         print('Epoch {}, training loss: {:.4f}'.format(i, loss_train.item()))

        self.eval()
            # output, x = self.forward(self.features, self.edge_index, self.edge_weight)
            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            # print("acc_val: {:.4f}".format(acc_val))
            # if acc_val > best_acc_val:
            #     best_acc_val = acc_val
        self.output = output

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output = self.forward(self.features, self.edge_index,self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids
# %%
