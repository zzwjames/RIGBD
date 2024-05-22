import torch 
from torch.distributions.bernoulli import Bernoulli
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy

import heuristic_selection as hs

class ABL(nn.Module):
    def __init__(self, args, model_name, model, isolation_ratio, threshold, loss_increse_threshold, isolate_epoch, device):
        super(ABL, self).__init__()
        self.args = args
        self.model_name = model_name
        self.model = model
        self.sur_model = deepcopy(self.model)
        self.device = device
        self.threshold = threshold
        self.loss_increse_threshold = loss_increse_threshold
        self.isolate_epoch = isolate_epoch
        self.isolation_ratio = isolation_ratio  # 0.01

    def compute_loss_value(self,idx_train):
        criterion = nn.NLLLoss(reduction = 'none').to(self.device)
        self.sur_model.eval()
        with torch.no_grad():
            output = self.forward_sur(self.features, self.edge_index, self.edge_weight)
            losses_record = criterion(output[idx_train],self.labels[idx_train])
        idx_losses_record = np.argsort(np.array(losses_record.detach().cpu()))
        
        # print(losses_record[-5:], losses_record[:-5])
        return idx_losses_record
    
    def isolate_data(self, idx_train):
        idx_losses_record = self.compute_loss_value(idx_train)
        idx_isolated = idx_train[idx_losses_record[0:int(len(idx_losses_record)*self.isolation_ratio)]]
        idx_clean = idx_train[idx_losses_record[int(len(idx_losses_record)*self.isolation_ratio):]]
        
        # print(idx_isolated)
        return idx_isolated, idx_clean
    
    def forward(self, features, edge_index, edge_weight):
        if(self.model_name == 'GCN'):
            output, x = self.model.forward(features, edge_index, edge_weight)
        elif(self.model_name in ['GAT','GraphSAGE','GIN']):
            output= self.model.forward(features, edge_index, edge_weight)
        else:
            raise NotImplementedError("No implemented model")
        return output

    def forward_sur(self, features, edge_index, edge_weight):
        if(self.model_name == 'GCN'):
            output, x = self.sur_model.forward(features, edge_index, edge_weight)
        elif(self.model_name in ['GAT','GraphSAGE','GIN']):
            output= self.sur_model.forward(features, edge_index, edge_weight)
        else:
            raise NotImplementedError("No implemented model")
        return output

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune=False, attach=None):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs

        """
        # assert 0 < prob_drop < 1, "prob_drop must be between 0 and 1 (exclusive)"
    
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)

        self.labels = labels.to(self.device)
        # self.prob_drop = prob_drop
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.model.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if verbose:
            print('=== training gnn model ===')
        optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr, weight_decay=self.model.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        isolate_epoch = 40

        for i in range(train_iters):

            self.model.train()
            if( i == isolate_epoch):
                idx_isolated, idx_clean = self.isolate_data(idx_train)
                
            if(i < isolate_epoch):
                threshold = self.threshold
                optimizer.zero_grad()
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train],reduction = 'none')
                # idx_isolated, idx_clean = self.isolate_data(idx_train)
                loss_train = torch.sign(loss_train - threshold) * loss_train
                loss_train.backward()
                optimizer.step()
            else:

                # rs_edge_index, rs_edge_weight = self.sample_noise_all(self.prob_drop, self.edge_index, self.edge_weight, self.device)

                output = self.forward(self.features, self.edge_index, self.edge_weight)
                optimizer.zero_grad()
                loss_train_isolated = -1 * F.nll_loss(output[idx_isolated], labels[idx_isolated])
                loss_train_isolated.backward()
                print('Epoch {}, loss_train_isolated: {}'.format(i, loss_train_isolated.item()))
                optimizer.step()

                optimizer.zero_grad()
                loss_train_clean = F.nll_loss(output[idx_clean], labels[idx_clean])
                # loss_train = loss_train_clean + loss_train_isolated
                # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train_clean.backward()
                print('Epoch {}, loss_train_clean: {}'.format(i, loss_train_clean.item()))
                optimizer.step()

                if verbose and i % 10 == 0:
                    print('Epoch {}, training loss: {}'.format(i, loss_train.item()))


        self.model.eval()
        output = self.forward(self.adv_x, self.adv_edge_index, self.adv_edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        
        if verbose:
            print('=== training gnn model ===')
        optimizer = optim.Adam(self.model.parameters(), lr=self.model.lr, weight_decay=self.model.weight_decay)
        optimizer_sur = optim.Adam(self.sur_model.parameters(), lr=self.model.lr, weight_decay=self.model.weight_decay)
        best_loss_val = 100
        best_acc_val = 0

        isolate_epoch = self.isolate_epoch

        for i in range(train_iters):

            # if( i == isolate_epoch):
            #     idx_isolated, idx_clean = self.isolate_data(idx_train)
                
            if(i < isolate_epoch):
                self.sur_model.train()
                self.model.eval()
                optimizer_sur.zero_grad()
                output = self.forward_sur(self.features, self.edge_index, self.edge_weight)
                loss_train = F.nll_loss(output[idx_train], labels[idx_train],reduction = 'none')
                # idx_isolated, idx_clean = self.isolate_data(idx_train)
                # print(loss_train[-31:])
                # print(loss_train[:-31])
                # loss_train = torch.sign(loss_train - self.threshold) * loss_train
                # # print(loss_train[-31:])
                loss_train = torch.mean(loss_train)
                loss_train.backward()
                optimizer_sur.step()
                idx_isolated, idx_clean = self.isolate_data(idx_train)
            else:
                self.sur_model.eval()
                self.model.train()

                # rs_edge_index, rs_edge_weight = self.sample_noise_all(self.prob_drop, self.edge_index, self.edge_weight, self.device)

                optimizer.zero_grad()
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_train_isolated = -1 * F.nll_loss(output[idx_isolated], labels[idx_isolated], reduction = 'none')
                loss_train_isolated = torch.relu(loss_train_isolated + self.loss_increse_threshold)
                loss_train_isolated = torch.mean(loss_train_isolated)

                # loss_train_isolated = -1 * loss_train_isolated
                loss_train_isolated.backward()
                print('Epoch {}, loss_train_isolated: {}'.format(i, loss_train_isolated.item()))
                optimizer.step()

                optimizer.zero_grad()
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_train_clean = F.nll_loss(output[idx_clean], labels[idx_clean])
                # loss_train = loss_train_clean + loss_train_isolated
                # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                loss_train_clean.backward()
                print('Epoch {}, loss_train_clean: {}'.format(i, loss_train_clean.item()))
                optimizer.step()

                # optimizer.zero_grad()
                # output = self.forward(self.features, self.edge_index, self.edge_weight)

                # loss_train_isolated = -1 * F.nll_loss(output[idx_isolated], labels[idx_isolated])
                # loss_train_clean = F.nll_loss(output[idx_clean], labels[idx_clean])
                # loss_train = loss_train_clean + loss_train_isolated

                # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
                # loss_train.backward()
                # optimizer.step()



                self.model.eval()
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = utils.accuracy(output[idx_val], labels[idx_val])
                
                if verbose and i % 10 == 0:
                    print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                    print("acc_val: {:.4f}".format(acc_val))
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.model.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.model.load_state_dict(weights)


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        
        self.model.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)

            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return float(acc_test)
    
    def attack_evaluation(self, args, x, edge_index, edge_weight, labels, idx_atk, idx_clean_test):
        output = self.forward(x, edge_index, edge_weight)
        train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
        print("ASR: {:.4f}".format(train_attach_rate))
        asr = train_attach_rate
        flip_idx_atk = idx_atk[(labels[idx_atk] != args.target_class).nonzero().flatten()]
        flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
        print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
        ca = self.test(x,edge_index, edge_weight,labels,idx_clean_test)
        print("CA: {:.4f}".format(ca))