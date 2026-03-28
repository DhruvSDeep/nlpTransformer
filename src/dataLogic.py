import pickle
import glob
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def wordFreqDict():
    freqDict={}
    txt_files = glob.glob("./data/**/*.txt", recursive=True)


    for i in txt_files:
        with open(i, 'r', encoding='utf-8', errors='ignore') as f:
            words=re.findall(r"\s*\w+|[^\w\s]", f.read())
            words = [w.replace(' ', '_') for w in words]
            for j in words:
                if j not in freqDict:
                    freqDict[j] = 1
                else:
                    freqDict[j] += 1
    with open("./data/frequencyDict.pkl", 'wb') as fout:
        pickle.dump(freqDict, fout)

def bytePairEncode(maxVocab):

    vocab = set()
    manual = []

    if not os.path.exists("./data/frequencyDict.pkl"):
        wordFreqDict()
    with open("./data/frequencyDict.pkl", 'rb') as f:
        freqDict = pickle.load(f)

    a = list(freqDict.keys())
    for i in a:
        freqDict[(tuple(i))] = freqDict[i]
        del freqDict[i]

    for i in freqDict:
        for j in i:
            vocab.add(j)


    # def occurenceCount(freqDict, substr):
    #     count = 0
    #     for word in freqDict:
    #         subcount=0
    #         for j in range(0, len(word)-1):
    #             if substr == word[j:j+2]:
    #                 subcount+=1
    #         count += subcount * freqDict[word]
    #     return(count)


    def loop(freqDict):
        pairs = {}
        for word, freq in freqDict.items():
            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        if not pairs:
            return ('', '', 0)
        best = max(pairs, key=pairs.get)
        return (best[0], best[1], pairs[best])
    
    def merge(a, b, freqDict):
        x = list(freqDict.keys())

        for word in x:
            if a in word:
                new_word = list(word)
                j=0
                while j < (len(new_word)-1):
                    if new_word[j] == a and new_word[j+1] == b:
                        new_word = new_word[:j] + [a+b] + new_word[j+2:]
                        j-=1
                    j+=1
                if tuple(new_word) != word:
                    freqDict[tuple(new_word)] = freqDict[word]
                    del freqDict[word]
        return(freqDict)
    
    for i in range (maxVocab - len(vocab)):
        maxSofar = loop(freqDict)
        if maxSofar[2] == 0:
            break
        freqDict =merge(maxSofar[0], maxSofar[1], freqDict)
        manual.append((maxSofar[0], maxSofar[1]))
        vocab.add(maxSofar[0]+maxSofar[1])

    return(manual, vocab)

def tokenise(stri, manual):
    words = re.findall(r"\s*\w+|[^\w\s]", stri)
    words = [w.replace(' ', '_') for w in words]
    
    merge_rank = {pair: i for i, pair in enumerate(manual)}
    
    result = []
    for word in words:
        tokens = list(word)
        while len(tokens) > 1:
            # find the best merge in this word
            best_pair = None
            best_rank = float('inf')
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j+1])
                if pair in merge_rank and merge_rank[pair] < best_rank:
                    best_rank = merge_rank[pair]
                    best_pair = pair
            if best_pair is None:
                break
            # apply that merge everywhere in the word
            a, b = best_pair
            j = 0
            while j < len(tokens) - 1:
                if tokens[j] == a and tokens[j+1] == b:
                    tokens[j] = a + b
                    tokens.pop(j+1)
                    j -= 1
                j += 1
        result.extend(tokens)
    return result




class tokenDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.samples = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            chunk = token_ids[i : i + seq_len + 1]
            self.samples.append(chunk)

    def __len__(self):
        return(len(self.samples))
    
    def __getitem__(self, idx):
        item = torch.tensor(self.samples[idx], dtype=torch.long)
        x = item[:-1]
        y = item[1:]
        return x, y



