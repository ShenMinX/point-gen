import torch
import torch.nn as nn

import random

from data import raw_data, Dataset, ixs_to_words
import torch.utils.data as data

from rouge import rouge_n_summary_level
from rouge import rouge_l_summary_level

from encoder import Encoder
from decoder import Decoder

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev) 

def my_collate(batch):
    sent = [torch.LongTensor(item[0]).to(device) for item in batch]
    target = [torch.LongTensor(item[1]).to(device) for item in batch]
    return [sent, target]

if __name__ == '__main__':
    
    # hyperparameters

    enc_embed_size = 128
    dec_embed_size = 128
    enc_hid_size = 256  # out_size = 200
    dec_hid_size = 256

    learning_rate = 0.0015

    lamada = 1  # weight of coverage loss

    epochs = 10

    tr_dict, tr_sents, tr_targets = raw_data(file_path = 'en\\pseudo_data.tsv') # pseudo_data.tsv
    train_data = Dataset(tr_sents, tr_targets, tr_dict.word_to_ix) 

    _, te_sents, te_targets = raw_data(file_path = 'en\\pseudo_data.tsv')
    test_data = Dataset(te_sents, te_targets, tr_dict.word_to_ix) # use train_dictionary!
    
    train_loader = data.DataLoader(dataset=train_data, batch_size=32, shuffle=False, collate_fn=my_collate)
    test_loader = data.DataLoader(dataset=test_data, batch_size=32, shuffle=False, collate_fn=my_collate)
    
    model_encoder = Encoder(
                      vocab=tr_dict.word_to_ix, 
                      hidden_size=enc_hid_size, 
                      embed_size=enc_embed_size
                      ).to(device)

    model_decoder = Decoder(
                      vocab=tr_dict.word_to_ix, 
                      encode_size=enc_hid_size*2, 
                      hidden_size=dec_hid_size, 
                      embed_size=dec_embed_size,
                      device=device
                     ).to(device)

    criterion = nn.NLLLoss()

    enc_optimizer = torch.optim.Adam(model_encoder.parameters(),lr=learning_rate)
    dec_optimizer = torch.optim.Adam(model_decoder.parameters(),lr=learning_rate)


    # train
    for e in range(epochs):
        total_loss = 0.0
        total_coverage_loss = 0.0

        for idx, item in enumerate(train_loader):

            enc_input, target = [i for i in item]

            enc_input = nn.utils.rnn.pad_sequence(enc_input, batch_first=True, padding_value=tr_dict.word_to_ix["<pad>"])

            target = nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=tr_dict.word_to_ix["<pad>"])

            max_sl = enc_input.shape[1]  # max sentence length
            max_tl = target.shape[1]     # max target length

            enc_out, enc_hidden = model_encoder(enc_input)

            batch_size = enc_input.shape[0]

            dec_input = torch.tensor([tr_dict.word_to_ix["<sos>"]]*batch_size, dtype=torch.long).to(device).view(batch_size, 1)

            with torch.no_grad():
                rnn_hid = (torch.zeros(batch_size,dec_hid_size).to(device),torch.zeros(batch_size,dec_hid_size).to(device)) # default init_hidden_value
                attn = torch.ones(batch_size, max_sl).to(device) # init_attn
                coverage = torch.zeros(batch_size, max_sl).to(device) # init_coverage
            
            
            batch_loss = 0.0
            for i in range(max_tl):

                output, coverage, rnn_hid, attn, coverage_loss = model_decoder(coverage, enc_out, rnn_hid, dec_input, enc_input, attn)
                
                _, dec_pred = torch.max(output, 1) # batch_size vector

                if random.randint(0, 11) > 5:          
                    dec_input = target[:,i].view(batch_size, 1)
                else:
                    dec_input = dec_pred.view(batch_size, 1)

                p_step_loss = -torch.log(-criterion(output, target[:,i]))

                batch_loss = batch_loss + p_step_loss + lamada*coverage_loss
            

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            batch_loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            with torch.no_grad():
                total_loss += float(batch_loss)
                total_coverage_loss += float(coverage_loss)
        print('%d: total loss= %f, total coverage loss= %f '% (e+1,total_loss, total_coverage_loss))
    
        

    # test
    
    with torch.no_grad():
        total_loss = 0.0
        total_coverage_loss = 0.0
        final_preds = []
        final_targets = []
        for idx, item in enumerate(test_loader):        
        
            enc_input, target = [i for i in item]

            enc_input = nn.utils.rnn.pad_sequence(enc_input, batch_first=True, padding_value=tr_dict.word_to_ix["<pad>"])

            target = nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=tr_dict.word_to_ix["<pad>"])

            max_sl = enc_input.shape[1]  # max sentence length
            max_tl = target.shape[1]     # max target length

            enc_out, enc_hidden = model_encoder(enc_input)

            batch_size = enc_input.shape[0]

            dec_input = torch.tensor([tr_dict.word_to_ix["<sos>"]]*batch_size, dtype=torch.long).to(device).view(batch_size, 1)

            rnn_hid = (torch.zeros(batch_size,dec_hid_size).to(device),torch.zeros(batch_size,dec_hid_size).to(device)) # default init_hidden_value
            
            attn = torch.ones(batch_size, max_sl).to(device) # init_attn
            coverage = torch.zeros(batch_size, max_sl).to(device) # init_coverage

            
            pred = torch.tensor([],dtype=torch.long).to(device)
            batch_loss = 0.0
            for i in range(max_tl):

                output, coverage, rnn_hid, attn, coverage_loss = model_decoder(coverage, enc_out, rnn_hid, dec_input, enc_input, attn)
                
                _, dec_pred = torch.max(output, 1) # batch_size vector

                pred = torch.cat([pred, dec_pred.view(batch_size, 1)], dim = 1)

                dec_input = dec_pred.view(batch_size, 1)

                p_step_loss = -torch.log(-criterion(output, target[:,i]))

                batch_loss = batch_loss + p_step_loss + lamada*coverage_loss

            
            total_loss += float(batch_loss)
            total_coverage_loss += float(coverage_loss)



        # unpad for evaluation
            for b in range(batch_size):
                final_pred = pred[b,:][pred[b,:]!=tr_dict.word_to_ix['<pad>']].tolist()
                final_preds.append(ixs_to_words(tr_dict.ix_to_word, final_pred))
                
        print('test set total loss: %f, total coverage loss: %f '% (total_loss, total_coverage_loss))

        print(final_preds[0])
        print(te_targets[0])# for pseudo_data, suppose output ['b','<eos>']      
        # dependency: easy-rouge 0.2.2, install: pip install easy-rouge
        _, _, rouge_1 = rouge_n_summary_level(final_preds, te_targets, 1)
        print('ROUGE-1: %f' % rouge_1)

        _, _, rouge_2 = rouge_n_summary_level(final_preds, te_targets, 2)
        print('ROUGE-2: %f' % rouge_2)
        
        # _, _, rouge_l = rouge_l_summary_level(final_preds, final_targets) # extremely time consuming...
        # print('ROUGE-L: %f' % rouge_l)
    

        

        

