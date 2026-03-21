import dataLogic
import transformerLogic
import torch
import pickle
from transformerLogic import (
    VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, EMBED_DIM, 
    NUM_HEADS, NUM_LAYERS, FF_DIM, LEARNING_RATE, EPOCHS
)
with open("./data/remap_dict", "rb") as f:
    remapping = pickle.load(f)

newVocabSize = len(remapping)
reverseReMap = dataLogic.reverse_remap()

seed = [1, 12, 16, 17, 12]


model = transformerLogic.transformer(newVocabSize, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
model.load_state_dict(torch.load("./checkpoints/model_weights.pt")) 

outputInt = transformerLogic.creation(model, seed)
for i in range(len(outputInt)):
    outputInt[i] = reverseReMap[outputInt[i]]

tokens = dataLogic.intToToken(outputInt)

dataLogic.detokenize_midi(tokens, "./outputs/trial_4.midi")