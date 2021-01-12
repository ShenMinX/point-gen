import torch
import torch.nn as nn
from data import raw_data, Dataset
import torch.utils.data as data

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size = 30, embed_size = 20):
        super(Encoder, self).__init__()
    
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embedding = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = embed_size, padding_idx=3)
        self.lstm = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs):
        
        embed = self.embedding(inputs)
        output, hidden = self.lstm(embed) 

        return output, hidden

if __name__ == "__main__":
    my_dict, sents, targets = raw_data()
    my_data = Dataset(sents, targets, my_dict.word_to_ix)
    loader = data.DataLoader(dataset=my_data, batch_size=32, shuffle=False)
    
    my_encoder = Encoder(len(my_dict.word_to_ix))
    print(len(sents), len(my_dict.word_to_ix))
    for idx, item in enumerate(loader):
        input, target = [i.type(torch.LongTensor) for i in item]
        print(input.shape)
        out, hidden = my_encoder(input)
        print(out.shape)
    
    