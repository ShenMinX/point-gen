import torch
import torch.nn as nn

import random

from data import raw_data, Dataset
import torch.utils.data as data

from rouge import rouge_n_summary_level
from rouge import rouge_l_summary_level

from encoder import Encoder
from decoder import Decoder


if __name__ == '__main__':
    
    # hyperparameters

    enc_embed_size = 128
    dec_embed_size = 128
    enc_hid_size = 256  # out_size = 200
    dec_hid_size = 256
    max_sl = 6  # max sentence length
    max_tl = 6  # max target length
    learning_rate = 0.0015

    epochs = 10

    tr_dict, tr_sents, tr_targets = raw_data(file_path = 'en\\pseudo_data.tsv')
    train_data = Dataset(tr_sents, tr_targets, tr_dict.word_to_ix, max_sl=max_sl, max_tl=max_tl)

    _, te_sents, te_targets = raw_data(file_path = 'en\\pseudo_data.tsv')
    test_data = Dataset(te_sents, te_targets, tr_dict.word_to_ix, max_sl=max_sl, max_tl=max_tl) # use train_dictionary!
    
    train_loader = data.DataLoader(dataset=train_data, batch_size=32, shuffle=False)
    test_loader = data.DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    
    model_encoder = Encoder(
                      vocab_size=len(tr_dict.word_to_ix), 
                      hidden_size=enc_hid_size, 
                      embed_size=enc_embed_size)

    model_decoder = Decoder(
                      vocab_size=len(tr_dict.word_to_ix), 
                      encode_size=enc_hid_size*2, 
                      hidden_size=dec_hid_size, 
                      embed_size=dec_embed_size
                     )

    criterion = nn.CrossEntropyLoss()

    enc_optimizer = torch.optim.Adam(model_encoder.parameters(),lr=learning_rate)
    dec_optimizer = torch.optim.Adam(model_decoder.parameters(),lr=learning_rate)


    # train
    for e in range(epochs):
        total_loss = 0.0

        for idx, item in enumerate(train_loader):
            
            enc_input, target = [i.type(torch.LongTensor) for i in item]

            enc_out, enc_hidden = model_encoder(enc_input)

            batch_size = enc_input.shape[0]

            dec_input = torch.tensor([tr_dict.word_to_ix["<sos>"]]*batch_size, dtype=torch.long).view(batch_size, 1)

            with torch.no_grad():
                rnn_hid = (torch.zeros(1,batch_size,dec_hid_size),torch.zeros(1,batch_size,dec_hid_size)) # default init_hidden_value
                attn = torch.ones(batch_size, max_sl) # init_attn
            
            
            batch_loss = 0.0
            for i in range(max_tl):

                output, rnn_hid, attn = model_decoder(enc_out, rnn_hid, dec_input, enc_input, attn)
                
                _, dec_pred = torch.max(output, 1) # batch_size vector

                if random.randint(0, 11) > 5:          
                    dec_input = target[:,i].view(batch_size, 1)
                else:
                    dec_input = dec_pred.view(batch_size, 1)

                p_step_loss = criterion(output, target[:,i])

                batch_loss = batch_loss + p_step_loss
            

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            batch_loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            with torch.no_grad():
                total_loss += batch_loss

        print('%d: total loss= %f'% (e+1,total_loss))
    
        

    # test
    with torch.no_grad():
        total_loss = 0.0
        preds = torch.tensor([],dtype=torch.long)
        targets = torch.tensor([],dtype=torch.long)
        
        for idx, item in enumerate(test_loader):        
        
            enc_input, target = [i.type(torch.LongTensor) for i in item]

            targets = torch.cat([targets, target], dim=0)

            enc_out, enc_hidden = model_encoder(enc_input)

            batch_size = enc_input.shape[0]

            dec_input = torch.tensor([tr_dict.word_to_ix["<sos>"]]*batch_size, dtype=torch.long).view(batch_size, 1)

            rnn_hid = (torch.zeros(1,batch_size,dec_hid_size),torch.zeros(1,batch_size,dec_hid_size)) # default init_hidden_value
            
            attn = torch.ones(batch_size, max_sl) # init_attn
            
            pred = torch.tensor([],dtype=torch.long)
            batch_loss = 0.0
            for i in range(max_tl):

                output,rnn_hid, attn = model_decoder(enc_out, rnn_hid, dec_input, enc_input, attn)
                
                _, dec_pred = torch.max(output, 1) # batch_size vector

                pred = torch.cat([pred, dec_pred.view(batch_size, 1)], dim = 1)

                dec_input = dec_pred.view(batch_size, 1)

                p_step_loss = criterion(output, target[:,i])

                batch_loss = batch_loss + p_step_loss 

            
            total_loss += batch_loss

            preds = torch.cat([preds,pred], dim=0)

        print('test set total loss: %f '% (total_loss))
        final_preds = []
        final_targets = []
        print(preds) # for pseudo_data, suppose output [[5, 1, 3, 3,...,3],...,[5, 1, 3, 3,...,3]]
        # unpad for evaluation
        for i in range(batch_size):
            final_pred = preds[i,:][preds[i,:]!=tr_dict.word_to_ix['<pad>']].tolist()
            final_preds.append(final_pred)
            final_target = targets[i,:][targets[i,:]!=tr_dict.word_to_ix['<pad>']].tolist()
            final_targets.append(final_target)

        # dependency: easy-rouge 0.2.2, install: pip install easy-rouge
        _, _, rouge_1 = rouge_n_summary_level(final_preds, final_targets, 1)
        print('ROUGE-1: %f' % rouge_1)

        _, _, rouge_2 = rouge_n_summary_level(final_preds, final_targets, 2)
        print('ROUGE-2: %f' % rouge_2)
        
        # _, _, rouge_l = rouge_l_summary_level(final_preds, final_targets) # extremely time consuming...
        # print('ROUGE-L: %f' % rouge_l)
    

        

        

