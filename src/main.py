import dataLogic
import transformerLogic
import pickle
import glob
from torch.utils.data import DataLoader
from torch import save
import os

from transformerLogic import (
    SEQ_LEN, BATCH_SIZE, EMBED_DIM, 
    NUM_HEADS, NUM_LAYERS, FF_DIM, LEARNING_RATE, EPOCHS
)

if (not os.path.exists("./data/manual.pkl")) or (not os.path.exists("./data/vocab.pkl")):
    manual, vocab = dataLogic.bytePairEncode(3000)
    with open("./data/manual.pkl", 'wb') as f:
        pickle.dump(manual, f)
    with open("./data/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
else:
    with open("./data/manual.pkl", 'rb') as f:
        manual = pickle.load(f)
    with open("./data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

vocab_list = sorted(vocab)
token_to_idx = {token: i + 3 for i, token in enumerate(vocab_list)}
token_to_idx['<PAD>'] = 0
token_to_idx['<SOS>'] = 1
token_to_idx['<EOS>'] = 2
VOCAB_SIZE = len(token_to_idx)

idx_to_token = {i: t for t, i in token_to_idx.items()}

txt_files = glob.glob("./data/**/*.txt", recursive=True)

tokens = []
for i in txt_files:
    with open(i, 'r', encoding='utf-8', errors='ignore') as f:

        tokens.extend(dataLogic.tokenise(f.read(), manual))

token_ids = [token_to_idx.get(t, token_to_idx['<PAD>']) for t in tokens]


with open("./data/token_to_idx.pkl", 'wb') as f:
    pickle.dump(token_to_idx, f)

dataset = dataLogic.tokenDataset(token_ids, seq_len=512)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



model = transformerLogic.transformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
transformerLogic.train(model, dataloader, EPOCHS, VOCAB_SIZE)
save(model.state_dict(), "./checkpoints/model_weights.pt")

