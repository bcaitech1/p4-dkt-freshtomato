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


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

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
        
        # lstm layer
        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

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

        categorical, continuous, _, __ = input
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
        
        # pass lstm layer
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        # pass last block hidden dimension to fully connected layer
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

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

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
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

        # pass lstm layer
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
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

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
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

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds



##################################
class GRUATTN(nn.Module):

    def __init__(self, args):
        super(GRUATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.encoder_layers = self.args.encoder_layers

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.embedding_dims = [self.hidden_dim // 3] * self.args.n_cates # cate feature 수와 동일하게 해주세요
        assert len(self.embedding_dims) == self.args.n_cates
        self.total_embedding_dims = sum(self.embedding_dims)
        # =========================================================================================================================

        self.cate_emb = nn.ModuleList([nn.Embedding(x, self.embedding_dims[idx]) for idx, x in enumerate(self.args.cate_embs)])
        self.cate_comb_proj = nn.Sequential(
            nn.Linear(self.total_embedding_dims, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2)
        )

        # continuous features
        self.cont_comb_proj = nn.Sequential(
            nn.Linear(self.args.n_conts, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2)
        )

        self.gru = nn.GRU(input_size=self.hidden_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.n_layers,
                           bidirectional=False,
                           dropout=self.drop_out,
                           batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.encoder_layers,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        return h

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

        # pass gru layer
        hidden = self.init_hidden(batch_size)
        out, hidden = self.gru(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.encoder_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


##############################################
class ADDGRUATTN(nn.Module):

    def __init__(self, args):
        super(ADDGRUATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.encoder_layers = self.args.encoder_layers

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.embedding_dims = [self.hidden_dim // 3] * self.args.n_cates # cate feature 수와 동일하게 해주세요
        assert len(self.embedding_dims) == self.args.n_cates
        self.total_embedding_dims = self.embedding_dims[0]
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

        self.gru = nn.GRU(input_size=self.hidden_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.n_layers,
                           bidirectional=False,
                           dropout=self.drop_out,
                           batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.encoder_layers,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        return h

    def forward(self, input):
        categorical, continuous, mask, __ = input
        batch_size = categorical[0].size(0)

        # categorical
        x_cat = [emb_layer(categorical[i]) for i, emb_layer in enumerate(self.cate_emb)]
        x_cat = sum(x_cat)
        x_cat = self.cate_comb_proj(x_cat)
        
        # continuous
        x_cont = torch.cat([c.unsqueeze(-1) for c in continuous], -1).to(torch.float32)
        x_cont = self.cont_comb_proj(x_cont)

        # concat Catgegorical & Continuous Feature
        X = sum([x_cat, x_cont])

        # pass gru layer
        hidden = self.init_hidden(batch_size)
        out, hidden = self.gru(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000
        head_mask = [None] * self.encoder_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


###################################################
class ATTNGRU(nn.Module):

    def __init__(self, args):
        super(ATTNGRU, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.encoder_layers = self.args.encoder_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.embedding_dims = [self.hidden_dim // 3] * self.args.n_cates # cate feature 수와 동일하게 해주세요
        assert len(self.embedding_dims) == self.args.n_cates
        self.total_embedding_dims = sum(self.embedding_dims)
        # =========================================================================================================================

        self.cate_emb = nn.ModuleList([nn.Embedding(x, self.embedding_dims[idx]) for idx, x in enumerate(self.args.cate_embs)])
        self.cate_comb_proj = nn.Sequential(
            nn.Linear(self.total_embedding_dims, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2)
        )

        # continuous features
        self.cont_comb_proj = nn.Sequential(
            nn.Linear(self.args.n_conts, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2)
        )

        self.gru = nn.GRU(input_size=self.hidden_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.n_layers,
                           batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.encoder_layers,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        return h

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
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.encoder_layers
        
        encoded_layers = self.attn(X, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        # pass gru layer
        hidden = self.init_hidden(batch_size)
        out, hidden = self.gru(sequence_output, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Saint(nn.Module):
    
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.drop_out
        
        ### Embedding 
        # ENCODER embedding
        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.embedding_dims = [self.hidden_dim // 3] * self.args.n_cates # cate feature 수와 동일하게 해주세요
        assert len(self.embedding_dims) == self.args.n_cates
        self.enc_embedding_dims = sum(self.embedding_dims[:-1])
        self.dec_embedding_dims = sum(self.embedding_dims)
        # =========================================================================================================================
        
        # categorical feature embedding for both encoder & decoder
        self.cate_emb = nn.ModuleList([nn.Embedding(x, self.embedding_dims[idx]) for idx, x in enumerate(self.args.cate_embs)])

        # ENCODER embedding
        # encoder combination projection
        self.enc_cate_comb_proj = nn.Sequential(
            nn.Linear(self.enc_embedding_dims, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # continuous features
        self.enc_cont_comb_proj = nn.Sequential(
            nn.Linear(self.args.n_conts, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.enc_proj = nn.Linear(self.hidden_dim*2 , self.hidden_dim)

        # DECODER embedding
        # decoder combination projection
        self.dec_cate_comb_proj = nn.Sequential(
            nn.Linear(self.dec_embedding_dims, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # continuous features
        self.dec_cont_comb_proj = nn.Sequential(
            nn.Linear(self.args.n_conts, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.dec_proj = nn.Linear(self.hidden_dim*2 , self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))


    def forward(self, input):

        categorical, continuous, mask, __ = input
        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # Encoder Embedding
        # categorical
        enc_x_cat = [emb_layer(categorical[i]) for i, emb_layer in enumerate(self.cate_emb[:-1])]
        enc_x_cat = torch.cat(enc_x_cat, -1)
        enc_x_cat = self.enc_cate_comb_proj(enc_x_cat)
        
        # continuous
        enc_x_cont = torch.cat([c.unsqueeze(-1) for c in continuous], -1).to(torch.float32)
        enc_x_cont = self.enc_cont_comb_proj(enc_x_cont)

        # concat
        embed_enc = torch.cat([enc_x_cat, enc_x_cont], -1)
        embed_enc = self.enc_proj(embed_enc)


        # DECODER Embedding
        # categorical
        dec_x_cat = [emb_layer(categorical[i]) for i, emb_layer in enumerate(self.cate_emb)]
        dec_x_cat = torch.cat(dec_x_cat, -1)
        dec_x_cat = self.dec_cate_comb_proj(dec_x_cat)

        # continuous
        dec_x_cont = torch.cat([c.unsqueeze(-1) for c in continuous], -1).to(torch.float32)
        dec_x_cont = self.dec_cont_comb_proj(dec_x_cont)

        # concat
        embed_dec = torch.cat([dec_x_cat, dec_x_cont], -1)
        embed_dec = self.dec_proj(embed_dec)
        
        
        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Saint_custom(nn.Module):
    
    def __init__(self, args):
        super(Saint_custom, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        # self.dropout = self.args.dropout
        self.dropout = 0.
        
        ### Embedding 
        # ENCODER embedding
        # categorical features
        # ========================== nn.Embedding에 들어갈 self.embedding_dims의 개수를 수정해주세요 ===============================
        self.embedding_dims = [self.hidden_dim // 3] * self.args.n_cates # cate feature 수와 동일하게 해주세요
        assert len(self.embedding_dims) == self.args.n_cates
        self.enc_embedding_dims = sum(self.embedding_dims[:-1])
        self.dec_embedding_dims = self.embedding_dims[-1]
        # =========================================================================================================================
        
        # categorical feature embedding for both encoder & decoder
        self.cate_emb = nn.ModuleList([nn.Embedding(x, self.embedding_dims[idx]) for idx, x in enumerate(self.args.cate_embs)])

        # ENCODER embedding
        # encoder combination projection
        self.enc_cate_comb_proj = nn.Sequential(
            nn.Linear(self.total_embedding_dims, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # continuous features
        self.enc_cont_comb_proj = nn.Sequential(
            nn.Linear(self.args.n_conts, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        self.enc_proj = nn.Linear(self.hidden_dim*2 , self.hidden_dim)

        # DECODER embedding
        # decoder combination projection
        self.dec_cate_comb_proj = nn.Sequential(
            nn.Linear(self.dec_embedding_dims, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    

    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))


    def forward(self, input):

        categorical, continuous, mask, __ = input
        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # Encoder Embedding (Exercise)
        # categorical
        enc_x_cat = [emb_layer(categorical[i]) for i, emb_layer in enumerate(self.cate_emb[:-1])]
        enc_x_cat = torch.cat(enc_x_cat, -1)
        enc_x_cat = self.enc_cate_comb_proj(enc_x_cat)
        
        # continuous
        enc_x_cont = torch.cat([c.unsqueeze(-1) for c in continuous], -1).to(torch.float32)
        enc_x_cont = self.enc_cont_comb_proj(enc_x_cont)

        # concat
        embed_enc = torch.cat([enc_x_cat, enc_x_cont], -1)
        embed_enc = self.enc_proj(embed_enc)


        # DECODER Embedding (Response = interaction)
        dec_interaction = self.cate_emb[-1](categorical[-1])
        dec_interaction = self.dec_cate_comb_proj(dec_interaction)

        embed_dec = dec_interaction
        
        
        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        
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

        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)      

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def get_mask(self, seq_len, mask, batch_size):
        new_mask = torch.zeros_like(mask)
        new_mask[mask == 0] = 1
        new_mask[mask != 0] = 0
        mask = new_mask
    
        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        categorical, continuous, mask, __ = input
        batch_size = categorical[0].size(0)
        seq_len = categorical[0].size(1)

        # categorical
        x_cat = [emb_layer(categorical[i]) for i, emb_layer in enumerate(self.cate_emb)]
        x_cat = torch.cat(x_cat, -1)
        x_cat = self.cate_comb_proj(x_cat)
        
        # continuous
        x_cont = torch.cat([c.unsqueeze(-1) for c in continuous], -1).to(torch.float32)
        x_cont = self.cont_comb_proj(x_cont)

        # concat Catgegorical & Continuous Feature
        embed = torch.cat([x_cat, x_cont], -1)
        embed = self.proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        self.mask = self.get_mask(seq_len, mask, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds
