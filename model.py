import sys
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score
from layers import Dense, CrossCompressUnit
import torch.nn.functional as F
from torch.autograd import Variable


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''
    """
    Revise
    1.  attn_view = self.softmax(attn)

        attn = self.dropout(attn_view)###
        
        output = torch.bmm(attn, v)

        return output, attn_view
    =>
        attn = self.softmax(attn)

        attn = self.dropout(attn)###
        
        output = torch.bmm(attn, v)

        return output, attn
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn_view = self.softmax(attn)
        attn = self.dropout(attn_view)
        output = torch.bmm(attn, v)
        return output, attn_view

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_v)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask != None : 
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
            output, attn = self.attention(q, k, v, mask=mask)

            output = output.view(n_head, sz_b, len_q, d_v)
            output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)
            return output, attn
        else:
            output, attn = self.attention(q, k, v, mask=mask)
            output = output.view(n_head, sz_b, len_q, d_v)
            output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
            output = self.dropout(self.fc(output))
            return output,attn  

class Test_model(nn.Module):
    def __init__(self, args, n_user, n_item, n_entity, n_relation):
        super(Test_model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.user_embeddings_lookup = nn.Embedding(self.n_user, self.args.dim).to(self.device)
        self.item_embeddings_lookup = nn.Embedding(self.n_item, self.args.dim).to(self.device)
        self.entity_embeddings_lookup = nn.Embedding(self.n_entity, self.args.dim).to(self.device)
        self.relation_embeddings_lookup = nn.Embedding(self.n_relation, self.args.dim).to(self.device)
        L_ = self.args.L_hgn
        d_ = self.args.dim

        self.feature_gate_item = nn.Linear(d_, d_).to(self.device)
        self.feature_gate_user = nn.Linear(d_, d_).to(self.device)

        self.instance_gate_item = Variable(torch.zeros(d_, 1).type(torch.FloatTensor),requires_grad=True).to(self.device)
        self.instance_gate_user = Variable(torch.zeros(d_, L_).type(torch.FloatTensor),requires_grad=True).to(self.device)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

        self.user_embeddings_lookup.weight.data.normal_(0, 1.0 / self.user_embeddings_lookup.embedding_dim)
        self.item_embeddings_lookup.weight.data.normal_(0, 1.0 / self.item_embeddings_lookup.embedding_dim)  
        self.entity_embeddings_lookup.weight.data.normal_(0, 1.0 / self.entity_embeddings_lookup.embedding_dim)  
        self.relation_embeddings_lookup.weight.data.normal_(0, 1.0 / self.relation_embeddings_lookup.embedding_dim)  
      
        self.W2 = nn.Embedding(self.n_item, d_, padding_idx=0).to(self.device)
        self.b2 = nn.Embedding(self.n_item, 1, padding_idx=0).to(self.device)      
        
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        # CNN Layer
        n_filters, filter_height = self.args.n_filters, self.args.filter_height

        self.horizontal_cnn_layer = nn.Conv2d(in_channels=1,out_channels=self.args.n_filters,kernel_size=(filter_height,d_),stride=1).to(self.device)
        self.vertical_cnn_layer = nn.Conv2d(in_channels=1,out_channels=self.args.n_filters,kernel_size=(L_,1),stride=1).to(self.device)
        
        # Maxpooling Layer
        self.maxpooling_layer = nn.MaxPool2d((1,L_-1), stride=1)

        # Fully Connected Layer
        self.W_FC_layer = Variable(torch.zeros(n_filters+(n_filters*d_), d_).type(torch.FloatTensor),requires_grad=True).to(self.device)
        self.W_FC_layer = torch.nn.init.xavier_uniform_(self.W_FC_layer)

        self.W_plun_FC_layer = Variable(torch.zeros(2*d_, d_).type(torch.FloatTensor),requires_grad=True).to(self.device)
        self.W_plun_FC_layer = torch.nn.init.xavier_uniform_(self.W_plun_FC_layer)

        self.tail_mlp = nn.Sequential()
        self.cc_unit = nn.Sequential()
        for i_cnt in range(self.args.L_mkr):
            self.tail_mlp.add_module('tail_mlp{}'.format(i_cnt),
                                     Dense(self.args.dim, self.args.dim,self.device))
            self.cc_unit.add_module('cc_unit{}'.format(i_cnt),
                                     CrossCompressUnit(self.args.dim,self.device))

        self.kge_pred_mlp = Dense(self.args.dim *2, self.args.dim,self.device)
        self.kge_mlp = nn.Sequential()
        for i_cnt in range(self.args.H_mkr-1):
            self.kge_mlp.add_module('kge_mlp{}'.format(i_cnt),
                                    Dense(self.args.dim *2, self.args.dim *2,self.device))

        self.W_r = Variable(torch.zeros(d_, d_).type(torch.FloatTensor),requires_grad=True).to(self.device)
        self.W_r = torch.nn.init.xavier_uniform_(self.W_r)
       
    def HGN_model(self,user_emb,item_embs,items_to_predict,for_pred):
        # GLM layer
        gate = torch.sigmoid(self.feature_gate_item(item_embs) + self.feature_gate_user(user_emb).unsqueeze(1))
        gated_item = item_embs * gate
        if torch.cuda.is_available():
            gated_item = torch.unsqueeze(gated_item, dim=1).type(torch.cuda.FloatTensor)#.type(torch.FloatTensor)
        else:
            gated_item = torch.unsqueeze(gated_item, dim=1).type(torch.FloatTensor)

        # CNN layer (H)
        self.c_h = self.horizontal_cnn_layer(gated_item)
        self.c_h = torch.squeeze(self.c_h)
        self.o_h = self.maxpooling_layer(self.c_h)
        self.o_h = torch.squeeze(self.o_h)
        # CNN layer (V)
        self.c_v = self.vertical_cnn_layer(gated_item)
        self.c_v = torch.squeeze(self.c_v)
        self.o_v = torch.flatten(self.c_v,1)
        #concat layer (H+V)
        self.o_h_v_concat = torch.cat((self.o_h,self.o_v),1)
        self.z = torch.matmul(self.o_h_v_concat, self.W_FC_layer.unsqueeze(0)).squeeze()
        self.u_z_concat = torch.cat((user_emb,self.z),1)
        
        union_out = torch.matmul(self.u_z_concat, self.W_plun_FC_layer.unsqueeze(0)).squeeze()
        w2 = self.W2(items_to_predict)
        b2 = self.b2(items_to_predict)

        if for_pred is not None:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            # MF
            res = user_emb.mm(w2.t()) + b2
            # union-level
            res += union_out.mm(w2.t())
            # item-item product
            rel_score = torch.matmul(item_embs, w2.t().unsqueeze(0))
            rel_score = torch.sum(rel_score, dim=1)
            res += rel_score
        else:
            # MF
            res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()
            # union-level
            res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
            # item-item product
            rel_score = item_embs.bmm(w2.permute(0, 2, 1))
            rel_score = torch.sum(rel_score, dim=1)
            res += rel_score
        return res

    def forward(self, input_):
        user_indices,item_indices = input_[0],input_[1]
        head_indices,relation_indices,tail_indices = input_[2],input_[3],input_[4]
        self.items_to_predict,self.for_pred = input_[5],input_[6]
        if user_indices is not None:
            self.user_indices = user_indices
        if item_indices is not None:
            self.item_indices = item_indices
        if head_indices is not None:
            self.head_indices = head_indices
        if relation_indices is not None:
            self.relation_indices = relation_indices
        if tail_indices is not None:
            self.tail_indices = tail_indices

        if user_indices is not None:  
            self.user_embeddings = self.user_embeddings_lookup(self.user_indices)
            self.item_embeddings = self.item_embeddings_lookup(self.item_indices)
            self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)

            seq_len = list(self.item_embeddings.size())[1]
            for i in range(seq_len):
                self.item_embeddings[:,i],self.head_embeddings[:,i]=self.cc_unit([self.item_embeddings[:,i],self.head_embeddings[:,i]])
            
            self.res = self.HGN_model(self.user_embeddings,self.item_embeddings,self.items_to_predict,self.for_pred)

            outputs = [self.user_embeddings,self.item_embeddings,self.res]
            if self.for_pred is not None:
                return outputs[-1]
            else:
                return outputs

        if self.relation_indices is not None:  #KG
            self.item_embeddings = self.item_embeddings_lookup(self.item_indices)
            self.head_embeddings = self.entity_embeddings_lookup(self.head_indices)
            self.item_embeddings, self.head_embeddings = self.cc_unit([self.item_embeddings, self.head_embeddings])
            self.tail_embeddings = self.entity_embeddings_lookup(self.tail_indices)
            self.relation_embeddings = self.relation_embeddings_lookup(self.relation_indices)
            self.Wr_t = torch.matmul(self.tail_embeddings, self.W_r.unsqueeze(0)).squeeze()
            self.Wr_h = torch.matmul(self.head_embeddings, self.W_r.unsqueeze(0)).squeeze()
            self.Wr_h_r = self.Wr_h + self.relation_embeddings
            self.tanh_Wr_h_r = nn.Tanh()(self.Wr_h_r)
            self.tanh_Wr_h_r_transpose = torch.t(self.tanh_Wr_h_r)
            self.Pi_hrt_part = torch.matmul(self.tanh_Wr_h_r_transpose,self.Wr_t.unsqueeze(0)).squeeze()
            self.Pi_hrt = torch.matmul(self.tail_embeddings,self.Pi_hrt_part.unsqueeze(0)).squeeze()
            self.tail_pred = torch.sum(self.Pi_hrt,0)
            self.scores_kge = torch.sigmoid(torch.sum(self.tail_embeddings * self.tail_pred, 1))
            self.rmse = torch.mean(
                torch.sqrt(torch.sum(torch.pow(self.tail_embeddings -
                           self.tail_pred, 2), 1) / self.args.dim))
            outputs = [self.head_embeddings, self.tail_embeddings, self.scores_kge, self.rmse]
            return outputs
