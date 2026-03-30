import pickle
import torch
import transformerLogic
import dataLogic
from transformerLogic import EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM

def generate(model, seed, manual, token_to_idx, idx_to_token):
    seedTokens = dataLogic.tokenise(seed, manual)
    seedIds = [token_to_idx.get(t, token_to_idx.get('<PAD>', 0)) for t in seedTokens]
    generated_ids = transformerLogic.creation(model, seed=seedIds, max_length=200, temperature=0.60, topK=30)
    tokens = [idx_to_token.get(idx, "") for idx in generated_ids]
    final_text = "".join(tokens).replace('_', ' ')
    
    return final_text



with open("./data/manual.pkl", 'rb') as f:
    manual = pickle.load(f)
with open("./data/token_to_idx.pkl", 'rb') as f:
    token_to_idx = pickle.load(f)
idx_to_token = {i: t for t, i in token_to_idx.items()}


VOCAB_SIZE = len(token_to_idx)

model = transformerLogic.transformer(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM)
model.load_state_dict(torch.load("./checkpoints/model_weights_of_epoch_{epoch}.pt"))

prompt = """
They never lie. But what they say is not the truth.
"""
output = generate(model, prompt, manual, token_to_idx, idx_to_token)
print(output)