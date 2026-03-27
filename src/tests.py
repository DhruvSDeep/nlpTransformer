import pickle

with open("./data/vocab.pkl", 'rb') as f:
    va = pickle.load(f)

print(va, len(va))