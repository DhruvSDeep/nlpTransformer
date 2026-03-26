import dataLogic
import transformerLogic
import pickle

manual, vocab = dataLogic.bytePairEncode(3000)
with open("./data/vocab.pkl", 'wb') as f1:
    pickle.dump(vocab, f1)
with open("./data/manual.pkl", 'wb') as f2:
    pickle.dump(manual, f2)