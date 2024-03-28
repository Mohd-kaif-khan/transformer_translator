import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PostionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self):
        even_i = torch.arange(0, self.d_model,2).float()
        denominator = torch.pow(10000, even_i/self.d_model)

        position = torch.arange(0,self.max_seq_len).reshape(self.max_seq_len,1)
        even_PE = torch.sin(position/denominator)
        odd_PE = torch.cos(position/denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)

        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        return PE


class SentenceEmbedding(nn.Module):
    def __init__(self, vocab, d_model, max_seq_len) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(len(vocab), d_model)
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.postion = PostionEncoding(d_model, max_seq_len)

    def batchTokenize(self, batch, start_token, end_token):
        def tokenize(sentnece, start_token, end_token):
            words = [self.vocab[w] for w in sentnece.split()]

            if start_token:
                words.insert(0, self.vocab['<SOS>'])
            if end_token:
                words.append(self.vocab['<EOS>'])
            for _ in range(len(words), self.max_seq_len):
                words.append(self.vocab['<POS>'])
            return torch.tensor(words)
        tokenized = []
        for ind in range(len(batch)):
            tokenized.append( tokenize(batch[ind], start_token, end_token) )

        return torch.stack(tokenized)

    def forward(self, x, start_token=False, end_token=False):
        x = self.batchTokenize(x, start_token, end_token)

        x = self.embedding(x)
        pos = self.postion()
        return (x+pos)



class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head, head_dim) -> None:
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_head = num_head

        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, seq_len , input_dim = x.size()
        q = self.q_layer(x).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)
        k = self.k_layer(x).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)
        v = self.v_layer(x).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)

        scaled = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scaled = scaled.permute(1,0,2,3) + mask
            scaled = scaled.permute(0,1,2,3)

        attention = F.softmax(scaled, dim=-1)
        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)
        attention = torch.sum(attention, dim =1)
        V = torch.matmul(attention, v).permute(0,2,1,3).reshape(batch_size, seq_len, self.num_head*self.head_dim)

        return self.linear(V)


class LayerNormalize(nn.Module):
    def __init__(self,parameter_shape, eps = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameter_shape))
        self.beta = nn.Parameter(torch.zeros(parameter_shape))
    
    def forward(self, x):
        dims=-1
        mean = x.mean(dim=dims, keepdim=True)
        var = ( (x-mean)**2 ).mean(dim=dims, keepdim=True)

        std = (var+self.eps).sqrt()

        out = (x-mean)/std

        return self.gamma*out+self.beta


class PositionWiseFeedForward(nn.Module):
    def __init__(self, ffn, drop_prob,d_model) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=drop_prob)
        self.layer1 = nn.Linear(d_model, ffn)
        self.layer2 = nn.Linear(ffn, ffn//2)
        self.layer3 = nn.Linear(ffn//2, ffn//4)
        self.layer4 = nn.Linear(ffn//4, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.layer4(x)
        return x
    

class EncoderLayers(nn.Module):
    def __init__(self, d_model, ffn, drop_prob, num_head, head_dim) -> None:
        super().__init__()
        self.selfAttention = MultiheadAttention(d_model, num_head, head_dim)
        self.norm = LayerNormalize([d_model])
        self.feedForward = PositionWiseFeedForward(ffn, drop_prob, d_model)
        self.linear = nn.Linear(d_model,d_model)

    def forward(self, x, mask):
        residual_x = x.clone()

        x = self.selfAttention(x, mask)
        x = self.norm(x+residual_x)

        residual_x = x.clone()

        x = self.feedForward(x)
        x = self.norm(residual_x+x)

        return self.linear(x)


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x , mask = inputs
        for layer in self._modules.values():
            x = layer(x,mask)
        return x




class Encoder(nn.Module):
    def __init__(self, d_model,ffn,drop_prob,max_seq_len,num_head,head_dim,eng_vocab,start_token,end_token,padding_token,n_layers) -> None:
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.sentence_embedding = SentenceEmbedding(eng_vocab, d_model,max_seq_len)
        self.enocder_layers = SequentialEncoder( *[EncoderLayers(d_model, ffn, drop_prob, num_head, head_dim) for _ in range(n_layers)] )
    
    def forward(self, x,mask):
        x = self.sentence_embedding(x,self.start_token, self.end_token)
        x = self.enocder_layers(x, mask)

        return x



class CrossMultiheadAttention(nn.Module):
    def __init__(self, d_model,num_head, head_dim) -> None:
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_head = num_head

        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    # def forward(self, x,y,mask):
    #     batch_size, seq_len, input_dim = x.size()
    #     q = self.q_layer(y).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)
    #     v = self.v_layer(x).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)
    #     k = self.k_layer(x).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)

    #     scaled = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(self.head_dim)

    #     if mask is not None:
    #         scaled = scaled.permute(1,0,2,3)+mask
    #         scaled = scaled.permute(1,0,2,3)

    #     attention = F.softmax(scaled, dim=-1)
    #     attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)
    #     attention = torch.sum(attention, dim =1)

    #     V = torch.matmul(attention,v).permute(0,2,1,3).reshape(batch_size, seq_len, self.head_dim*self.num_head)

    #     return self.linear(V)
    
    def forward(self, x, y,mask):
        batch_size, seq_len , input_dim = x.size()
        q = self.q_layer(y).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)
        k = self.k_layer(x).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)
        v = self.v_layer(x).reshape(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)

        scaled = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.head_dim)

        if mask is not None:
            scaled = scaled.permute(1,0,2,3) + mask
            scaled = scaled.permute(0,1,2,3)

        attention = F.softmax(scaled, dim=-1)

        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)
        attention = torch.sum(attention, dim =1)

        V = torch.matmul(attention, v).permute(0,2,1,3).reshape(batch_size, seq_len, self.num_head*self.head_dim)

        return self.linear(V)


class DecoderLayers(nn.Module):
    def __init__(self, d_model, ffn, drop_prob, num_head, head_dim) -> None:
        super().__init__()
        self.cross_attention = CrossMultiheadAttention(d_model,num_head, head_dim)
        self.selfAttention = MultiheadAttention(d_model, num_head, head_dim)
        self.norm = LayerNormalize([d_model])
        self.feedForward = PositionWiseFeedForward(ffn, drop_prob, d_model)
        self.linear = nn.Linear(d_model,d_model)

    def forward(self, x,y,cross_attention_mask, self_attention_mask):
        residual_y = y.clone()
        y = self.selfAttention(y , self_attention_mask)
        y = self.norm(residual_y+y)

        residual_y = y.clone()
        y = self.cross_attention(x,y,cross_attention_mask)
        y = self.norm(residual_y+y)

        residual_y = y.clone()
        y = self.feedForward(y)
        y = self.norm(residual_y+y)

        return y



# class SequentialDecoder(nn.Sequential):
#     def forward(self,*inputs):
#         x,y, attention_mask,cross_attention_mask = input

#         for layer in self._modules.values():
#             y = layer(x, y, attention_mask, cross_attention_mask)

#         return y


class SequentialDecoder(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs

        for layer in self.layers:
            y = layer(x, y, self_attention_mask, cross_attention_mask)
        return y



class Decoder(nn.Module):
    def __init__(self, d_model,ffn,drop_prob,max_seq_len,num_head,head_dim,hn_vocab,start_token,end_token,padding_token,n_layers) -> None:
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.sentence_embedding = SentenceEmbedding(hn_vocab, d_model, max_seq_len)
        self.decoder_layers = SequentialDecoder( *[DecoderLayers(d_model, ffn, drop_prob, num_head, head_dim) for _ in range(n_layers)] )


    def forward(self,x,y,decoder_attention_mask,decoder_cross_attention_mask):
        y = self.sentence_embedding(y, self.start_token, self.end_token)
        y = self.decoder_layers(x,y,decoder_cross_attention_mask, decoder_attention_mask)

        return y


class Transformer(nn.Module):
    def __init__(self, d_model, max_seq_len, ffn, drop_prob, eng_vocab, hn_vocab,num_head, head_dim, n_layers, start_token, end_token, padding_token) -> None:
        super().__init__()
        self.encoder = Encoder(d_model,ffn,drop_prob,max_seq_len,num_head,head_dim,eng_vocab,start_token,end_token,padding_token,n_layers)
        self.decoder = Decoder(d_model,ffn,drop_prob,max_seq_len,num_head,head_dim,hn_vocab,start_token,end_token,padding_token,n_layers)
        self.linear = nn.Linear(d_model, len(hn_vocab))

    def forward(self, x,y,encoder_attention_mask, decoder_cross_attention_mask, decoder_attention_mask):

        x = self.encoder(x , encoder_attention_mask)
        y = self.decoder(x,y,decoder_attention_mask,decoder_cross_attention_mask)

        y = self.linear(y)

        out = F.softmax(y, dim=-1)

        return out