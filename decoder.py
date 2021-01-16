import torch
import torch.nn as nn



class Attention(nn.Module):
    def __init__(self, encode_size = 60, hidden_size = 30):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.wh = nn.Linear(encode_size, hidden_size)
        self.ws = nn.Linear(hidden_size, hidden_size)
        self.wc = nn.Parameter(torch.rand(hidden_size),requires_grad=True)

        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)

    def forward(self, coverage, enc_out, rnn_out):

        #enc_out: batch x sen_len x encode_size
        #rnn_out: batch x 1 x hidden_size
        batch_size = enc_out.size(0)
        sen_len = enc_out.size(1)
        encode_size = enc_out.size(2)
        hidden_size = self.hidden_size

        wh_applied = self.wh(enc_out.reshape(-1, encode_size))  # batch*sen_len x hidden_size, reshape() instead of view() to fix a weird runtime error...
        wh_applied = wh_applied.reshape(batch_size, -1, hidden_size) # batch x sen_len x hidden_size

        ws_applied = self.ws(rnn_out.reshape(-1, hidden_size))  # batch*1 x hidden_size
        ws_applied = ws_applied.reshape(batch_size, -1, hidden_size) #batch x 1 x hidden_size

        coverage_transformed = coverage.reshape(-1, 1) # batch x sen_len -> batch*sen_len x 1
        wc_applied = coverage_transformed * self.wc.reshape(1, -1)
        wc_applied = wc_applied.reshape(batch_size, -1, hidden_size) # batch x sen_len x hidden_size

        energy = torch.matmul(torch.tanh(wh_applied + ws_applied + wc_applied), self.v) # batch x sen_len

        attn = torch.softmax(energy, 1)

        return attn

class Decoder(nn.Module):
    def __init__(self, vocab, encode_size = 60, hidden_size = 30, embed_size = 20):
        super(Decoder, self).__init__()
        
        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = embed_size, padding_idx=vocab["<pad>"])

        self.lstm = nn.LSTMCell(input_size = embed_size + encode_size, hidden_size = hidden_size)

        self.attn = Attention(encode_size = encode_size, hidden_size=hidden_size)

        self.wh = nn.Parameter(torch.rand(encode_size), requires_grad = True)
        self.ws = nn.Parameter(torch.rand(hidden_size), requires_grad = True)
        self.wx = nn.Parameter(torch.rand(embed_size), requires_grad = True)

        self.v = nn.Linear(hidden_size, self.vocab_size)
    
    def forward(self, coverage, enc_out, rnn_hid, dec_input, enc_inputs, attn):
        
        batch_size = enc_out.size(0)
        encode_size = enc_out.size(2)
        sen_len = enc_out.size(1)
        decode_size = self.hidden_size

        attn_transformed = attn.view(batch_size, -1, sen_len) # batch x sen_len -> batch x 1 x sen_len
        context = torch.bmm(attn_transformed, enc_out) # batch x 1 x encode_size

        embed = self.embedding(dec_input) # input: batch x 1 
    
        ctxt_embed = torch.cat([context, embed], 2).view(batch_size, -1)

        h_t, c_t = self.lstm(ctxt_embed, rnn_hid)

        rnn_out = h_t

        rnn_hid = (h_t, c_t)

        p_vocab = torch.softmax(self.v(rnn_out), 1) # batch x hidden_size -> batch x vocab_size
        p_gen = torch.sigmoid(torch.matmul(context, self.wh) + torch.matmul(rnn_out.view(batch_size, 1, -1), self.ws) + torch.matmul(embed, self.wx)) # batch x 1

        attn = self.attn(coverage, enc_out, rnn_out)

        coverage_loss = torch.min(attn, coverage)
        coverage_loss = torch.sum(coverage_loss)

        coverage = coverage + attn
        
        attn_scores = torch.zeros(batch_size, self.vocab_size)
        attn_scores = attn_scores.scatter_(1, enc_inputs, attn) # index: enc_inputs, content: attn 

        output = p_gen*p_vocab + (1-p_gen)*attn_scores # batch x vocab_size


        return output, coverage, rnn_hid, attn, coverage_loss


        

if __name__=="__main__":
    # batch_size = 5, encode_size = 6, hidden_size = 3, sen_len = 7, vocab_size = 12
    coverage = torch.zeros(5, 7)
    enc_out = torch.rand(5, 7, 6)
    rnn_out = torch.rand(5, 1, 3)

    my_attn = Attention(encode_size=6, hidden_size=3)
    attn = my_attn(coverage, enc_out, rnn_out)
    print(attn.size())
    
    vocab_size = 12
    encode_inputs = torch.LongTensor(5, 7).random_(0, vocab_size)
    decoder_input = torch.ones(5, 1, dtype=torch.long)





        


        

