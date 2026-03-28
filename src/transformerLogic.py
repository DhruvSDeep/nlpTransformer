import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import save

SEQ_LEN = 512
BATCH_SIZE = 32
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FF_DIM = 1024
LEARNING_RATE = 3e-4
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
class positional_encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        posEn = torch.zeros(SEQ_LEN, embed_dim)
        position = torch.arange(0, SEQ_LEN).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        posEn[:, 0::2] = torch.sin(position * div_term)
        posEn[:, 1::2] = torch.cos(position * div_term)
        posEn = posEn.unsqueeze(0)
        self.register_buffer('pe', posEn)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class multiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  #splits into our multiple heads
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) #this is for query/key matching
        if mask is not None:            #this does the triangle matrix. mask is a matrix with 1s everywhere, and 0s above the diag. 
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) #This recombines multiple heads into 1
        return self.dropout(self.out_proj(out))


class feedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.expand = nn.Linear(embed_dim, ff_dim)
        self.gelu = nn.GELU()
        self.compress = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x=self.expand(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.compress(x)
        return self.dropout(x)

class transformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = multiHeadAttention(embed_dim, num_heads)
        self.feed_fwd = feedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        attn = self.attention(self.norm1(x), mask)
        x = x + attn
        fedFwd = self.feed_fwd(self.norm2(x))
        x = x + fedFwd
        return x
    
class transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim):
        super().__init__()
        self.tokenEmbedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = positional_encoder(embed_dim)

        self.layers = nn.ModuleList([transformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.outHead = nn.Linear(embed_dim, vocab_size, bias = False)
        self.outHead.weight = self.tokenEmbedding.weight
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask = None):
        x = self.tokenEmbedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.outHead(x)
        return logits
    
def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
#this create the triangle matrix for our mask, with zeroes above diag



def train(model, dataloader, epochs, vocab_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Training on {device}, {sum(p.numel() for p in model.parameters()):,} parameters")
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    mask = create_causal_mask(SEQ_LEN, device)

    total_steps = epochs * len(dataloader)
    warpUpStep = 500
    def lr_lambda(currStep):
        if currStep < warpUpStep:
            return (currStep + 1) / warpUpStep
        else:
            return 0.5 * (1 + math.cos(math.pi * (currStep - warpUpStep) / (total_steps - warpUpStep)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for x, y in dataloader:
            x=x.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(x, mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
            scheduler.step()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}") 
        
   

        if epoch % 10 == 0 and epoch > 1:
            save(model.state_dict(), "./checkpoints/model_weights_interrupted.pt")
    
    
def creation(model, seed, max_length=512, temperature=1, topK=50):
    device = next(model.parameters()).device
    model.eval()

    generated = seed.copy()
    with torch.no_grad():
        for _ in range (max_length):
            context = generated[-SEQ_LEN:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            mask = create_causal_mask(x.size(1), device)
            logits = model(x, mask)
            logits = logits[0,-1,:]/temperature

            if topK > 0:
                values, indices = torch.topk(logits, topK)
                logits[logits < values[-1]] = float('-inf')

            probabs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabs, 1).item()
            if next_token == 2:
                break
            generated.append(next_token)
        return generated

