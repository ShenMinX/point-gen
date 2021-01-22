import re
import csv
import numpy as np
import torch
# from rouge import rouge_n_summary_level
# from rouge import rouge_l_summary_level


import torch.utils.data as data



class dictionary():

    def __init__(self):
        self.word_to_ix = {}
        self.ix_to_word = {}
        self.insert_mark()
    
    def insert(self, word):
        if word not in self.word_to_ix:
            ix = len(self.word_to_ix)
            self.word_to_ix[word] = ix
            self.ix_to_word[ix] = word
    
    def insert_mark(self):
        marks = ["<pad>","<sos>","<eos>","<UNK>"]
        for m in marks:
            self.insert(m)

#turn a list of words to list of indices  
def words_to_ixs(word_to_ix, words):
    out = []
    for w in words:
        if w in word_to_ix:
            out.append(word_to_ix[w])
        else:
            out.append(word_to_ix["<UNK>"])
    return out

def ixs_to_words(ix_to_word, ixs):
    out = []
    for w in ixs:
        out.append(ix_to_word[w])
    return out
    

def raw_data(file_path = 'en\\train.tsv'):

    my_dictionary = dictionary()
    sents = []
    targets = []
    fieldnames = ['id','sentsence1','sentsence2','label']
    with open(file_path, "r", encoding="utf-8") as tsv_file:
        reader = csv.DictReader(tsv_file, fieldnames=fieldnames, delimiter="\t")
        for i, row in enumerate(reader):
            if i ==0:
                continue
            
            if row['label'] == '1':
                #sent = re.split(" ", row['sentsence1'])
                sent = [w.group(0).lower() for w in re.finditer(r"\S+", row['sentsence1'])]
                sent.append('<eos>') # add 'end of sentence' mark
                sents.append(sent)
                for w1 in sent:
                    my_dictionary.insert(w1.lower())

                #target = re.split(" ", row['sentsence2'])
                target = [w.group(0).lower() for w in re.finditer(r"\S+", row['sentsence2'])]
                target.append('<eos>') # add 'end of sentence' mark
                targets.append(target)
                for w2 in target:
                    my_dictionary.insert(w2.lower())

            elif row['label'] != '1':
                continue
                    
    return my_dictionary, sents, targets



class Dataset(data.Dataset):

    def __init__(self, sents, targets, word_to_ix): # , max_sl = 200, max_tl = 200
        'Initialization'
        self.sents = sents
        self.targets = targets
        self.word_to_ix = word_to_ix
        #padding_index = 0
        self.pad = word_to_ix['<pad>']

    
    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        
        sent = self.sents[index]
        target = self.targets[index]

        sent, target = list(map(lambda x: words_to_ixs(self.word_to_ix, x),[sent, target]))
        # sent = np.array(sent)
        # target = np.array(target)

        return (sent, target)

    def padding(self, sent, max_len): # outdated
        sen_pad = np.pad(sent,(0,max(0, max_len - len(sent))),'constant', constant_values = (self.pad))[:max_len]
        return sen_pad

def my_collate(batch):
    sent = [torch.LongTensor(item[0]) for item in batch]
    target = [torch.LongTensor(item[1]) for item in batch]
    return [sent, target]
    

if __name__ == "__main__":
    my_dict, sents, targets = raw_data()
    my_data = Dataset(sents, targets, my_dict.word_to_ix)
    loader = data.DataLoader(dataset=my_data, batch_size=32, shuffle=False, collate_fn=my_collate)
    max = 0
    for s, t in zip(sents, targets):
        if len(s)>max:
            max = len(s)
            max_sen = s
        if len(t)>max:
            max = len(t)
            max_sen = t
    print(max)
    print(max_sen)

    # _, _, rouge_1 = rouge_n_summary_level(sents, targets, 1)
    # print('ROUGE-1: %f' % rouge_1)
    # print(sents[0])
    

    # _, _, rouge_1x = rouge_n_summary_level(torch.zeros(5, 3, dtype=torch.long).tolist(),torch.zeros(5, 3, dtype=torch.long).tolist(), 1)
    # print('ROUGE-1: %f' % rouge_1x)

    # _, _, rouge_2 = rouge_n_summary_level(sents, targets, 2)
    # print('ROUGE-2: %f' % rouge_2)

    # _, _, rouge_l = rouge_l_summary_level(sents, targets)
    # print('ROUGE-L: %f' % rouge_l)
    
