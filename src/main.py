import transformerLogic
import dataLogic
import glob
import os
from torch.utils.data import DataLoader
from torch import save

VOCAB_SIZE = 128 * 31 * 64 + 32 + 3   # notes + rests + special tokens
SEQ_LEN = 512
BATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FF_DIM = 1024
LEARNING_RATE = 3e-4
EPOCHS = 50

dataPaths =glob.glob("./data/**/*.midi", recursive=True)
data = []
for i in range(len(dataPaths)):
    try:
        data.append(dataLogic.tokenToInt(dataLogic.tokenize_midi(dataPaths[i])))
    except:
        print('1 file failed')

dataset = dataLogic.MidiDataset(data, seq_len=512)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)




model = transformerLogic.transformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
transformerLogic.train(model, dataloader, EPOCHS)
save(model.state_dict(), "./checkpoints/model_weights.pt")

    