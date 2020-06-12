from model import Test_model
import torch
import numpy as np
import random
from interactions import *
import pickle


class Train:
    def __init__(self,param_,data_,test_,data_process):
        self.args,self.n_user,self.n_item = param_[0],param_[1],param_[2]
        self.n_entity,self.n_relation = param_[3],param_[4]
        self.users_np,self.sequences_np,self.targets_np = data_[0],data_[1],data_[2]
        self.train_matrix = data_[3]
        self.test_ = test_
        self.data_process = data_process
        self.model_()
        self.optimizer_()

    def model_(self):
        self.test_model = Test_model(self.args,self.n_user,self.n_item,self.n_entity,self.n_relation)
        self.device = self.test_model.device

    def optimizer_(self):
        self.optimizer_rs = torch.optim.Adam(self.test_model.parameters(),lr=self.args.lr_rs)
        self.optimizer_kge = torch.optim.Adam(self.test_model.parameters(),lr=self.args.lr_kge)

    def l2_loss(self, inputs):
        return torch.sum(inputs ** 2) / 2

    def loss_rs_(self,loss_rs_input_hgn,loss_rs_input_mkr):
        targets_prediction,negatives_prediction = loss_rs_input_hgn[0],loss_rs_input_hgn[1]
        user_embeddings,item_embeddings = loss_rs_input_mkr[0],loss_rs_input_mkr[1]
        #hgn part
        loss_hgn = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
        loss_hgn = torch.mean(torch.sum(loss_hgn))
        #mkr part
        l2_loss_mkr = self.l2_loss(user_embeddings)+self.l2_loss(item_embeddings)
        for name, param in self.test_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('rs' in name) or ('cc_unit' in name) or ('user' in name)) \
                    and ('weight' in name):
                l2_loss_mkr = l2_loss_mkr + self.l2_loss(param)
        loss_rs = loss_hgn + l2_loss_mkr * self.args.l2_weight
        return loss_rs

    def loss_kge_(self, scores_kge, head_embeddings, tail_embeddings):
        base_loss_kge = -scores_kge
        l2_loss_kge = self.l2_loss(head_embeddings) + self.l2_loss(tail_embeddings)
        for name, param in self.test_model.named_parameters():
            if param.requires_grad and ('embeddings_lookup' not in name) \
                    and (('kge' in name) or ('tail' in name) or ('cc_unit' in name)) \
                    and ('weight' in name):
                l2_loss_kge = l2_loss_kge + self.l2_loss(param)
        # Note: L2 regularization will be done by weight_decay of pytorch optimizer
        loss_kge = base_loss_kge + l2_loss_kge * self.args.l2_weight
        return loss_kge, base_loss_kge, l2_loss_kge
    
    def negsamp_vectorized_bsearch_preverif(self,pos_inds, n_items, n_samp=32):
        raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
        pos_inds_adj = pos_inds - np.arange(len(pos_inds))
        neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
        return neg_inds

    def generate_negative_samples(self,train_matrix, num_neg=3, num_sets=10):
        neg_samples = []
        for user_id, row in enumerate(train_matrix):
            pos_ind = row.indices
            neg_sample = self.negsamp_vectorized_bsearch_preverif(pos_ind, train_matrix.shape[1], num_neg * num_sets)
            neg_samples.append(neg_sample)

        return np.asarray(neg_samples).reshape(num_sets, train_matrix.shape[0], num_neg)

    def Train_RS(self):
        self.test_model.train()
        n_train = self.sequences_np.shape[0]
        record_indexes = np.arange(n_train)
        np.random.shuffle(record_indexes)
        batch_size = self.args.batch_size
        num_batches = int(n_train / batch_size) + 1

        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size
            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break
            batch_record_index = record_indexes[start:end]

            batch_users = self.users_np[batch_record_index]
            batch_sequences = self.sequences_np[batch_record_index]
            batch_targets = self.targets_np[batch_record_index]

            negatives_np_multi = self.generate_negative_samples(self.train_matrix, self.args.neg_samples, self.args.sets_of_neg_samples)
            negatives_np = negatives_np_multi[batchID % self.args.sets_of_neg_samples]
            batch_neg = negatives_np[batch_users]

            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(self.device)
            batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(self.device)
            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)

            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(self.device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(self.device)

            rs_batch_data = [batch_users,batch_sequences,batch_sequences,None,None,items_to_predict,None]
            output_ = self.test_model(rs_batch_data)
            user_embeddings,item_embeddings,prediction_score = output_[0],output_[1],output_[2]

            (targets_prediction, negatives_prediction) = torch.split(
                prediction_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)
            
            loss_rs_input_hgn = [targets_prediction,negatives_prediction]
            loss_rs_input_mkr = [user_embeddings,item_embeddings]
            self.loss_rs = self.loss_rs_(loss_rs_input_hgn,loss_rs_input_mkr)
            self.epoch_loss += self.loss_rs.item()
            self.optimizer_rs.zero_grad()
            self.loss_rs.backward()
            self.optimizer_rs.step()
            self.epoch_loss /= num_batches
            self.loss_rs.detach()
            user_embeddings.detach()
            item_embeddings.detach()
            prediction_score.detach()
        return self.loss_rs,self.epoch_loss

    def Train_KGE(self,kg):
        batch_size_,batch_size_kge = 1,32
        self.test_model.train()
        h_set = list(set(kg.keys()))
        n_train = len(h_set)
        record_indexes = np.arange(n_train)
        np.random.shuffle(record_indexes)
        num_batches = int(n_train / batch_size_) + 1

        for batchID in range(num_batches):
            start = batchID * batch_size_
            end = start + batch_size_

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break
            batch_record_index = record_indexes[start]
            batch_head = h_set[batch_record_index]
            
            batch_relation = kg[batch_head][:,1]
            batch_tail = kg[batch_head][:,2]
            batch_head = np.array(batch_head).reshape((1))

            batch_head = torch.from_numpy(batch_head).to(self.device)
            batch_relation = torch.from_numpy(batch_relation).type(torch.LongTensor).to(self.device)
            batch_tail = torch.from_numpy(batch_tail).type(torch.LongTensor).to(self.device)

            kge_batch_data = [None,batch_head,batch_head,batch_relation,batch_tail,None,None]
            output_ = self.test_model(kge_batch_data)
            head_embeddings,tail_embeddings = output_[0],output_[1]
            scores_kge,rmse = output_[2],output_[3]

            loss_kge,base_loss_kge,l2_loss_kge = self.loss_kge_(scores_kge,head_embeddings,tail_embeddings)
        
            self.optimizer_kge.zero_grad()
            loss_kge.sum().backward()
            self.optimizer_kge.step()
            loss_kge.detach()
            head_embeddings.detach()
            tail_embeddings.detach()
            scores_kge.detach()
            rmse.detach()
        return rmse,loss_kge
 
    def evaluation(self,model_, test_, topk=20,save=False):
        train,test_sequences,test_Y_set,uid_attribute_category = test_[0],test_[1],test_[2],test_[3]
        num_users = train.num_users
        num_items = train.num_items
        batch_size = 1024
        num_batches = int(num_users / batch_size) + 1
        user_indexes = np.arange(num_users)
        item_indexes = np.arange(num_items)
        pred_list = 1
        train_matrix = train.tocsr()

        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < num_users:
                    end = num_users
                else:
                    break

            batch_user_index = user_indexes[start:end]

            batch_test_sequences = test_sequences[batch_user_index]
            batch_test_sequences = np.atleast_2d(batch_test_sequences)

            batch_test_sequences = torch.from_numpy(batch_test_sequences).type(torch.LongTensor).to(self.device)
            
            item_ids = torch.from_numpy(item_indexes).type(torch.LongTensor).to(self.device)
            batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(self.device)

            input_ = [batch_user_ids,batch_test_sequences,batch_test_sequences,None,None,item_ids,pred_list]

            rating_pred = model_(input_)

            rating_pred = rating_pred.cpu().data.numpy().copy()
            rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

            ind = np.argpartition(rating_pred, -topk)
            ind = ind[:, -topk:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

            if batchID == 0:
                pred_list = batch_pred_list
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
        precision, recall, MAP, ndcg = [], [], [], []
        for k in [5, 10, 15, 20]:
            precision.append(precision_at_k(test_Y_set, pred_list, k))
            recall.append(recall_at_k(test_Y_set, pred_list, k))
            MAP.append(mapk(test_Y_set, pred_list, k))
            ndcg.append(ndcg_k(test_Y_set, pred_list, k))
        return precision, recall, MAP, ndcg

    def run(self,show_loss=True):
        self.epoch_loss = 0.0
        for epoch in range(self.args.n_epochs):
            print("Train RS")
            loss,epoch_loss = self.Train_RS()
            if show_loss:
                print(epoch_loss)
            if epoch % self.args.kge_interval == 0:
                print("Train KGE")
                rmse, loss_kge = self.Train_KGE(self.data_process.KG_random_generator())
                if show_loss:
                    print(rmse.cpu().data.numpy())
            if (epoch + 1) % 5 == 0 :
                self.test_model.eval()
                precision, recall, MAP, ndcg = self.evaluation(self.test_model, self.test_, topk=20,save=False)
                print('precision : ',precision)
                print('recall : ',recall)
                print('MAP : ',MAP)
                print('NDCG : ',ndcg)
            if (epoch + 1) == self.args.n_epochs:
                self.test_model.eval()
                precision, recall, MAP, ndcg = self.evaluation(self.test_model, self.test_, topk=20,save=True)
                print('precision : ',precision)
                print('recall : ',recall)
                print('MAP : ',MAP)
                print('NDCG : ',ndcg)
