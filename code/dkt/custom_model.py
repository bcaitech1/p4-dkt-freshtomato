import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )

class BaseConv(nn.Module):
    def __init__(self, args):
        super(BaseConv, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        self.pooling_layer = nn.AvgPool1d(kernel_size=self.hidden_dim)

        # Fully connected layer
        # self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)

        conv3_output = conv3_output.permute(0, 2, 1)
        pooling = self.pooling_layer(conv3_output)
        
        # out = self.fc(pooling)
        pred = self.activation(pooling).view(batch_size, -1)

        return pred
    

class DeepConv(nn.Module):
    def __init__(self, args):
        super(DeepConv, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim//2,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim//2, 1)
        self.activation = nn.Sigmoid()
        
        # Initialization
        torch.nn.init.xavier_normal_(self.conv1[0].weight)
        torch.nn.init.xavier_normal_(self.conv2[0].weight)
        torch.nn.init.xavier_normal_(self.conv3[0].weight)
        torch.nn.init.xavier_normal_(self.conv4[0].weight)
        torch.nn.init.xavier_normal_(self.fc.weight)
        

    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv1_output = conv1_output.permute(0, 2, 1)

        avg_output = self.avg_pool(conv1_output)
        avg_output = avg_output.permute(0, 2, 1)

        conv2_output = self.conv2(avg_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv3(conv3_output)

        res = conv2_output + conv4_output
        res = res.permute(0, 2, 1)
        max_output = self.max_pool(res)

        out = self.fc(max_output)
        pred = self.activation(out).view(batch_size, -1)

        return pred


class DeepConvRes3(nn.Module):
    def __init__(self, args):
        super(DeepConvRes3, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim//2,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
        self.fc = nn.Linear(self.hidden_dim//2, 1)
        self.activation = nn.Sigmoid()
        
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)
    
    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv1_output = conv1_output.permute(0, 2, 1)

        avg_output = self.avg_pool(conv1_output)
        avg_output = avg_output.permute(0, 2, 1)
        
        conv1_output = conv1_output.permute(0, 2, 1)
        
        conv2_output = self.conv2(avg_output) + conv1_output
        conv3_output = self.conv3(conv2_output) + conv2_output
        conv4_output = self.conv3(conv3_output)

        res = conv3_output + conv4_output
        res = res.permute(0, 2, 1)
        max_output = self.max_pool(res)

        out = self.fc(max_output)
        pred = self.activation(out).view(batch_size, -1)

        return pred


class DeepConvAttn(nn.Module):
    def __init__(self, args):
        super(DeepConvAttn, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=self.drop_out),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim//2,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=self.drop_out),
            nn.ReLU()
        )
        
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=self.drop_out),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=self.drop_out),
            nn.ReLU()
        )

        # Fully connected layer
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        
        self.activation = nn.Sigmoid()

    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv1_output = conv1_output.permute(0, 2, 1)

        avg_output = self.avg_pool(conv1_output)
        avg_output = avg_output.permute(0, 2, 1)

        conv2_output = self.conv2(avg_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv3(conv3_output)

        res = conv2_output + conv4_output
        res = res.permute(0, 2, 1)

        out = res.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        pred = self.activation(out).view(batch_size, -1)

        return pred

    
class DeepConvAttnVer2(nn.Module):
    def __init__(self, args):
        super(DeepConvAttnVer2, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                    padding_mode='zeros'),
            nn.ELU()
        )
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim//2,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.ELU()
        )
        
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.ELU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.ELU()
        )

        # Fully connected layer
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=self.drop_out)
        
        self.activation = nn.Sigmoid()

    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv1_output = conv1_output.permute(0, 2, 1)

        avg_output = self.avg_pool(conv1_output)
        avg_output = avg_output.permute(0, 2, 1)

        conv2_output = self.conv2(avg_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv3(conv3_output)

        res = conv2_output + conv4_output
        res = res.permute(0, 2, 1)

        out = res.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)
        out = self.dropout(out)
        pred = self.activation(out).view(batch_size, -1)

        return pred


class DeepConvWithAttnBlock(nn.Module):
    def __init__(self, args):
        super(DeepConvWithAttnBlock, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=self.hidden_dim,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim//2,
                               out_channels=self.hidden_dim,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        
        self.conv3 = nn.Conv1d(in_channels=self.hidden_dim,
                               out_channels=self.hidden_dim,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        self.conv4 = nn.Conv1d(in_channels=self.hidden_dim,
                               out_channels=self.hidden_dim,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        self.mha = MultiHeadAttention(self.n_heads, self.hidden_dim)
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=self.drop_out)
        
        self.activation = nn.Sigmoid()

    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv1_attention = self.mha(conv1_output)
        conv1_attention = F.elu(conv1_attention)
        conv1_attention = conv1_attention.permute(0, 2, 1)

        avg_output = self.avg_pool(conv1_attention)
        avg_output = avg_output.permute(0, 2, 1)

        conv2_output = self.conv2(avg_output)
        conv2_attention = self.mha(conv2_output)
        conv2_attention = F.elu(conv2_attention)
        
        conv3_output = self.conv3(conv2_attention)
        conv3_attention = self.mha(conv3_output) + conv2_output
        conv3_attention = F.elu(conv3_attention)
        
        conv4_output = self.conv3(conv3_attention)
        conv4_attention = self.mha(conv4_output) + conv3_output
        res = F.elu(conv4_attention)
        res = res.permute(0, 2, 1)
        
        out = res.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        out = self.dropout(out)
        
        pred = self.activation(out).view(batch_size, -1)

        return pred


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
    
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -np.inf)
        
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output
    
    def forward(self, conv_output, mask=None):
        conv_output = conv_output.permute(0, 2, 1)
        
        bs = conv_output.size(0)
        
        # perform linear operation and split into h heads
        k = self.k_linear(conv_output).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(conv_output).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(conv_output).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        
        output = self.out(concat)
        output = output.permute(0, 2, 1)
        
        return output
    

class DeepConvRes(nn.Module):
    def __init__(self, args):
        super(DeepConvRes, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim//2,
                    out_channels=self.hidden_dim,
                    kernel_size=3, 
                    padding=1,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=5, 
                    padding=2,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=7, 
                    padding=3,
                    padding_mode='zeros'),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim//2, 1)
        self.dropout = nn.Dropout(p=self.drop_out)
        
        self.activation = nn.Sigmoid()

    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv1_output = conv1_output.permute(0, 2, 1)

        avg_output = self.avg_pool(conv1_output)
        avg_output = avg_output.permute(0, 2, 1)

        conv2_output = self.conv2(avg_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv3(conv3_output)

        res = conv2_output + conv4_output
        res = res.permute(0, 2, 1)
        max_output = self.max_pool(res)
        
        out = self.fc(max_output)
        pred = self.activation(out).view(batch_size, -1)

        return pred


class DeepConvRes2(nn.Module):
    def __init__(self, args):
        super(DeepConvRes2, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.in_channels = self.args.n_cates + self.args.n_conts
        # =========================================================================================================================

        self.conv1 = nn.Conv1d(in_channels=self.in_channels,
                               out_channels=self.hidden_dim//8,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim//16,
                               out_channels=self.hidden_dim//8,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        
        self.conv3 = nn.Conv1d(in_channels=self.hidden_dim//8,
                               out_channels=self.hidden_dim//4,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        self.conv4 = nn.Conv1d(in_channels=self.hidden_dim//4,
                               out_channels=self.hidden_dim//2,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        self.conv5 = nn.Conv1d(in_channels=self.hidden_dim//2,
                               out_channels=self.hidden_dim,
                               kernel_size=3, 
                               padding=1,
                               padding_mode='zeros')
        
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim//2, 1)
        self.dropout = nn.Dropout(p=self.drop_out)
        
        self.activation = nn.Sigmoid()

    def forward(self, input):
        categorical, continuous, mask, __ = input

        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # concat Catgegorical & Continuous Feature
        x_cat = torch.stack(categorical) # [17, 128, 250]
        x_cont = torch.stack(continuous) # [3, 128, 250]

        embed = torch.cat([x_cat, x_cont], 0) # [20, 128, 250]
        embed = embed.permute(1, 0, 2)   # [128, 20, 250]

        conv1_output = self.conv1(embed.type(torch.FloatTensor).to(self.device))
        conv1_output = conv1_output.permute(0, 2, 1)

        avg_output = self.avg_pool(conv1_output)
        avg_output = avg_output.permute(0, 2, 1)
        
        conv1_output = conv1_output.permute(0, 2, 1)
        conv2_output = self.conv2(avg_output) + conv1_output
        conv3_output = self.conv3(conv2_output) + conv2_output
        conv4_output = self.conv3(conv3_output)

        res = conv3_output + conv4_output
        res = res.permute(0, 2, 1)
        max_output = self.max_pool(res)
        
        out = self.fc(max_output)
        pred = self.activation(out).view(batch_size, -1)

        return pred

#####################################################################

class kobert_Classifier(nn.Module):
    def __init__(self, num_classes=42):
        super(kobert_Classifier, self).__init__()
        
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.embedding_dims = [self.hidden_dim // 3] * self.args.n_cates # cate feature 수와 동일하게 해주세요
        assert len(self.embedding_dims) == self.args.n_cates
        self.total_embedding_dims = sum(self.embedding_dims)
        # =========================================================================================================================

        self.cate_emb = nn.ModuleList([nn.Embedding(x, self.embedding_dims[idx]) for idx, x in enumerate(self.args.cate_embs)])
        
        self.cate_comb_proj = nn.Sequential(
            nn.Linear(self.total_embedding_dims, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # continuous features
        self.cont_comb_proj = nn.Sequential(
            nn.Linear(self.args.n_conts, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.proj = nn.Linear(self.hidden_dim*2 , self.hidden_dim)
        
        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)
        self.dr_rate = self.args.drop_out
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        out = self.bert(input_ids=token_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        
        if self.dr_rate:
            out = self.dropout(out)
        
        return self.classifier(out)
    

    
'''
Reference:
    https://github.com/monologg/R-BERT
'''
class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class RBert(nn.Module):
    def __init__(self, args):
        super(RBert, self).__init__()
        
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.embedding_dims = [self.hidden_dim // 3] * self.args.n_cates # cate feature 수와 동일하게 해주세요
        assert len(self.embedding_dims) == self.args.n_cates
        self.total_embedding_dims = sum(self.embedding_dims)
        # =========================================================================================================================

        self.cate_emb = nn.ModuleList([nn.Embedding(x, self.embedding_dims[idx]) for idx, x in enumerate(self.args.cate_embs)])
        
        self.cate_comb_proj = nn.Sequential(
            nn.Linear(self.total_embedding_dims, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # continuous features
        self.cont_comb_proj = nn.Sequential(
            nn.Linear(self.args.n_conts, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.proj = nn.Linear(self.hidden_dim*2 , self.hidden_dim)
        
        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )
        
        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)
        
        self.target_fc = FCLayer(self.hidden_dim, self.hidden_dim//2, 0.0)
        self.answer_fc = FCLayer(self.hidden_dim, self.hidden_dim//2, 0.0)
        self.label_classifier = FCLayer(self.hidden_dim//2 * 3, 1, 0.0, False)
        
        self.activation = nn.Sigmoid()

    def forward(self, input):
        categorical, continuous, mask, __ = input
        batch_size = categorical[0].size(0)
        
        # categorical
        x_cat = [emb_layer(categorical[i]) for i, emb_layer in enumerate(self.cate_emb)]
        x_cat = torch.cat(x_cat, -1)
        x_cat = self.cate_comb_proj(x_cat)
        
        # continuous
        x_cont = torch.cat([c.unsqueeze(-1) for c in continuous], -1).to(torch.float32)
        x_cont = self.cont_comb_proj(x_cont)

        # concat Catgegorical & Continuous Feature
        X = torch.cat([x_cat, x_cont], -1)
        X = self.proj(X)
        
        pr_answer = categorical[-1][:, 1:]
        positive = torch.where(pr_answer == 2)[1]
        negative = torch.where(pr_answer == 1)[1]
        
        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        enc_out = encoded_layers[0]
        
        target_vector = enc_out[:, -1, :] # 마지막 vector
        positive_vector = enc_out[:, positive, :]
        negative_vector = enc_out[:, negative, :]
        
        positive_vector = torch.mean(positive_vector, dim=1) # Average
        negative_vector = torch.mean(negative_vector, dim=1)
        
        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        target_embedding = self.target_fc(target_vector)
        positive_embedding = self.answer_fc(positive_vector)
        negative_embedding = self.answer_fc(negative_vector)
        
        # Concat -> fc_layer
        concat_embedding = torch.cat([target_embedding, positive_embedding, negative_embedding], dim=-1)
        
        out = self.label_classifier(concat_embedding)
        
        preds = self.activation(out).view(batch_size, -1)
        
        return preds