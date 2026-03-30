import dataLogic
import transformerLogic
import pickle
import glob
from torch.utils.data import DataLoader
import torch
import os

from transformerLogic import (
    SEQ_LEN, BATCH_SIZE, EMBED_DIM, 
    NUM_HEADS, NUM_LAYERS, FF_DIM, LEARNING_RATE, EPOCHS
)

randomNum = .58

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

if (not os.path.exists("./data/tokenizedFiles.pkl")) or (not os.path.exists("./data/valTokenizedFiles.pkl")):
    txt_files = glob.glob("./data/**/*.txt", recursive=True)

    tokens = []
    valTokens = []
    for i in txt_files:
        with open(i, 'r', encoding='utf-8', errors='ignore') as f:
            tokens_forAbook = dataLogic.tokenise(f.read(), manual)
            lenbook = len(tokens_forAbook)
            valTokens_forAbook = tokens_forAbook[int(randomNum * lenbook):int((randomNum+.05) * lenbook)]
            trainTokens = tokens_forAbook[:int(randomNum * lenbook)]
            trainTokens.extend(tokens_forAbook[int((randomNum+.05) * lenbook):])
            tokens.extend(trainTokens)
            valTokens.extend(valTokens_forAbook)

    token_ids = [token_to_idx.get(t, token_to_idx['<PAD>']) for t in tokens]
    val_token_ids = [token_to_idx.get(t, token_to_idx['<PAD>']) for t in valTokens]
    with open("./data/tokenizedFiles.pkl", 'wb') as f:
        pickle.dump(token_ids, f)
    with open("./data/valTokenizedFiles.pkl", 'wb') as f:
        pickle.dump(val_token_ids, f)
else:
    with open("./data/tokenizedFiles.pkl", 'rb') as f:
        token_ids = pickle.load(f)
    with open("./data/valTokenizedFiles.pkl", 'rb') as f:
        val_token_ids = pickle.load(f)
print(f"Training on {len(token_ids):,} tokens")
with open("./data/token_to_idx.pkl", 'wb') as f:
    pickle.dump(token_to_idx, f)

dataset = dataLogic.tokenDataset(token_ids, seq_len=512)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
validSet = dataLogic.tokenDataset(val_token_ids, seq_len=512)
validLoader = DataLoader(validSet, batch_size=BATCH_SIZE, shuffle=True)

model = transformerLogic.transformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
# model.load_state_dict(torch.load("./checkpoints/model_weights_of_epoch_{epoch}.pt"))
transformerLogic.train(model, dataloader, validLoader, EPOCHS, VOCAB_SIZE)
torch.save(model.state_dict(), "./checkpoints/model_weights.pt")

