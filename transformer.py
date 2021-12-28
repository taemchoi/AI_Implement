import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_length).unsqueeze(1)
        p_encoding = torch.zeros(max_length, d_model)
        p_encoding[:,0::2]= np.sin(position/np.power(10000, (np.arange(0, d_model, 2)/d_model)))
        p_encoding[:,1::2]= np.cos(position/np.power(10000, (np.arange(0, d_model, 2)/d_model)))
        p_encoding = p_encoding.unsqueeze(0).float()
        self.register_buffer('pe', p_encoding)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1),:]
        
def get_attention_pad_mask(query, key):
    batch_size, sequence = key.size()
    pad_mask = sequence==0
    pad_mask=pad_mask.unsqueeze(1).expand(batch_size, query, key)
    return pad_mask

class Scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super(Scaled_dot_product_attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def __call__(self, q, k, v, mask=None):
        d_k = k.size(-1)
        attention_key = torch.matmul(q,k.transpose(-2,-1))/np.sqrt(d_k) # batch_size, n_head, sequence, sequence
        if mask:
            attention_key=attention_key.masked_fill_(mask, 1e-12) # batch_size, n_head, sequence, sequence
        attention_score = torch.matmul(self.softmax(attention_key), v)  # batch_size, n_head, sequence, d_model/n_head
        return attention_score

class Multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(Multi_head_attention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.scaled_attention = Scaled_dot_product_attention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_cat = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_dim(self, batch_size, x):
        """[summary]
        Args:
            batch_size ([int]): 배치사이즈
            x ([tensor]): input tensor, x.shpae : batch, sequence, d_model

        Returns:
            [tensor]: batch_Size, n_head, sequence, d_model/n_head
        """
        out = x.view((batch_size,  -1,  self.n_head,  int(self.d_model/self.n_head)))   
        return out.transpose(1,2)     

    def __call__(self, q,k,v, mask=False):
        batch_size = q.size(0)
        query = self.split_dim(batch_size, self.w_q(q)) # batch_Size, n_head, sequence, d_model/n_head
        key = self.split_dim(batch_size, self.w_k(k)) # batch_Size, n_head, sequence, d_model/n_head
        value = self.split_dim(batch_size, self.w_v(v)) # batch_Size, n_head, sequence, d_model/n_head
        
        out = self.scaled_attention(query,key,value, mask=mask) # batch_Size, n_head, sequence, d_model/n_head
        out = out.transpose(1, 2) # batch_Size, sequence, n_head, d_model/n_head
        out = out.contiguous().view(batch_size, -1, self.d_model)  # batch_Size, sequence, d_model
        out = self.w_cat(out) # batch_Size, sequence, d_model
        return self.layer_norm((out+q))     # batch_Size, sequence, d_model
        
class Position_wise_feed_forward(nn.Module):
    def __init__(self, d_model, drop_out):
        super(Position_wise_feed_forward,self).__init__()
        self.fc_1=nn.Linear(d_model, 2048)
        self.fc_2=nn.Linear(2048, d_model)
        self.relu=nn.ReLU()
        self.drop_out=nn.Dropout(p= drop_out)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def __call__(self, x):
        out = self.fc_2(self.drop_out(self.relu(self.fc_1(x)))) #  batch_Size, sequence, d_model
        return self.layer_norm((out+x))  #  batch_Size, sequence, d_model
        
class Encoder_layer(nn.Module):
    def __init__(self, d_model, n_head, drop_out):
        super(Encoder_layer, self).__init__()
        self.enc_attention=Multi_head_attention(d_model, n_head)
        self.pff_network = Position_wise_feed_forward(d_model, drop_out)
        
    def __call__(self, x, enc_mask):
        """

        Args:
            x ([Tensor]): positional embedding까지 완료된 tensor(batch, sequence, d_model)

        Returns:
            out ([Tensor]): encoder의 output tensor(batch, sequence, d_model) 
        """
        out = self.enc_attention(q=x, k=x, v=x, mask=enc_mask)
        out = self.pff_network(out)
        return out, enc_mask
    
class Encoder(nn.Module):
    def __init__(self,num_encoder_layers, d_model, n_head, drop_out):
        super(Encoder,self).__init__()
        self.encoder_layers = nn.Sequential()
        for i in range(num_encoder_layers):
            self.encoder_layers.add_module(f'encoder{i}', Encoder_layer(d_model, n_head, drop_out)) 
    
    def __call__(self, x, enc_mask):
        out = self.encoder_layers(x, enc_mask)
        return out
    
class Decoder_layer(nn.Module):
    def __init__(self, d_model, n_head, drop_out):
        super(Decoder_layer, self).__init__()
        self.masked_dec_attention=Multi_head_attention(d_model, n_head)
        self.enc_dec_attention=Multi_head_attention(d_model, n_head)
        self.pff_network = Position_wise_feed_forward(d_model, drop_out)
        
    def __call__(self, encoder_output, decoder_input, dec_mask, enc_dec_mask):
        output = self.masked_dec_attention(q=decoder_input,k=decoder_input,v=decoder_input, mask=dec_mask)
        output = self.enc_dec_attention(q=output, k=encoder_output, v=encoder_output, mask=enc_dec_mask)
        output = self.pff_network(output)
        return encoder_output, output, dec_mask, enc_dec_mask


class Decoder(nn.Module):
    def __init__(self,  d_model, n_head, drop_out, num_decoder_layers):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.Sequential()
        for i in range(num_decoder_layers):
            self.decoder_layers.add_module(f'encoder{i}', Decoder_layer(d_model, n_head, drop_out)) 
        
    def __call__(self, enc_ouput, dec_input, enc_dec_mask, dec_mask):
        output = self.decoder_layers(enc_ouput, dec_input, dec_mask, enc_dec_mask)
        return output
        
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, num_encoder_layers, d_model, n_head, drop_out, num_decoder_layers, target_vocab_size, max_length):
        super(Transformer, self).__init__()
        self.enc = Encoder(num_encoder_layers, d_model, n_head, drop_out)
        self.dec = Decoder(d_model, n_head, drop_out, num_decoder_layers) 
        # self.embed = preprocessing_fc_layer(input_feature=input_feature, d_model=d_model, num_category=num_category, embedding_dim=embedding_dim)
        self.embed = nn.Embedding(input_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_length)
        self.fc_1 = nn.Linear(d_model, target_vocab_size)
        self.softmax = nn.SoftMax(target_vocab_size)
        
    def __call__(self, enc_input, dec_input, category):
        enc_mask = get_attention_pad_mask(enc_input, enc_input)
        dec_mask = torch.gt((get_attention_pad_mask(dec_input, dec_input) + get_attn_subsequent_mask(dec_input), 0))
        enc_dec_mask = get_attention_pad_mask(dec_input, enc_input) 
        embed_enc_input = self.embed(enc_input, category)
        enc_output = self.enc(embed_enc_input, enc_mask)
        dec_input = self.embed(dec_input, category)
        dec_output = self.dec(enc_output, dec_input, enc_dec_mask, dec_mask)
        output = self.Softmax(self.fc_1(dec_output))
        return output 