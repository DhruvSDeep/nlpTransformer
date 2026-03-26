import pickle
import glob
import os

def wordFreqDict():
    freqDict={}
    txt_files = glob.glob("./data/**/*.txt", recursive=True)

    for filepath in txt_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            clean_text = f.read()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(clean_text)

    for i in txt_files:
        with open(i, 'r', encoding='utf-8') as f:
            words=(f.read()).split()
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


                    


    