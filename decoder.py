import torch
import torch.nn as nn



class Attention(nn.Module):
    def __init__(self, encode_size = 60, hidden_size = 30):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.wh = nn.Linear(encode_size, hidden_size)
        self.ws = nn.Linear(hidden_size, hidden_size)

        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)

    def forward(self, enc_out, rnn_out):

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


        energy = torch.matmul(torch.tanh(wh_applied + ws_applied), self.v) # batch x sen_len

        attn = torch.softmax(energy, 1)

        return attn

class Decoder(nn.Module):
    def __init__(self, vocab, encode_size = 60, hidden_size = 30, embed_size = 20):
        super(Decoder, self).__init__()
        
        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = embed_size, padding_idx=vocab['<pad>'])

        self.lstm = nn.LSTM(input_size = embed_size + encode_size, hidden_size = hidden_size, bidirectional=False, batch_first=True)

        self.attn = Attention(encode_size = encode_size, hidden_size=hidden_size)

        self.wh = nn.Parameter(torch.rand(encode_size), requires_grad = True)
        self.ws = nn.Parameter(torch.rand(hidden_size), requires_grad = True)
        self.wx = nn.Parameter(torch.rand(embed_size), requires_grad = True)

        self.v = nn.Linear(hidden_size, self.vocab_size)
    
    def forward(self, enc_out, rnn_hid, dec_input, enc_inputs, attn):
        
        batch_size = enc_out.size(0)
        encode_size = enc_out.size(2)
        sen_len = enc_out.size(1)
        decode_size = self.hidden_size

        attn_transformed = attn.view(batch_size, -1, sen_len) # batch x sen_len -> batch x 1 x sen_len
        context = torch.bmm(attn_transformed, enc_out) # batch x 1 x encode_size

        embed = self.embedding(dec_input) # input: batch x 1 
    
        ctxt_embed = torch.cat([context, embed], 2)

        rnn_out, rnn_hid = self.lstm(ctxt_embed, rnn_hid)

        unnormalized_out = self.v(rnn_out.squeeze())

        attn = self.attn(enc_out, rnn_out)
        
        attn_scores = torch.zeros(batch_size, self.vocab_size)
        attn_scores = attn_scores.scatter_(1, enc_inputs, attn) # index: enc_inputs, content: attn 

# batch x vocab_size

        return unnormalized_out, rnn_hid, attn


        

if __name__=="__main__":
    # batch_size = 5, encode_size = 6, hidden_size = 3, sen_len = 7, vocab_size = 12

    enc_out = torch.rand(5, 7, 6)
    rnn_out = torch.rand(5, 1, 3)

    my_attn = Attention(encode_size=6, hidden_size=3)
    attn = my_attn(enc_out, rnn_out)
    print(attn.size())
    
    vocab_size = 12
    encode_inputs = torch.LongTensor(5, 7).random_(0, vocab_size)
    decoder_input = torch.ones(5, 1, dtype=torch.long)

    my_decoder = Decoder(vocab_size=vocab_size, encode_size=6, hidden_size=3, embed_size=5)

    rnn_hid = (torch.zeros(1,5,3),torch.zeros(1,5,3)) # default init_value

    output, rnn_hid, attn = my_decoder( enc_out, rnn_hid, decoder_input, encode_inputs, attn)
    output, rnn_hid, attn = my_decoder(enc_out, rnn_hid, decoder_input, encode_inputs, attn)
    print(output.size(),  attn.size())
    print(output)
    print(attn)



        


        
