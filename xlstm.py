# xlstm.py (fixed & robust)
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from math import floor
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

# ConvLSTM Cell

class ConvLSTMCell(nn.Module):    # look at current frane , hidden frame and produces new 
    def __init__(self, input_dim, hidden_dim, kernel_size=3):  #hidden dim -> means no.of channels in hidden state(h_t) and cell state (c_t)
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim,4 * hidden_dim,kernel_size,padding=padding,bias=True)

    def forward(self, x, h_prev, c_prev):
        # x: (B, input_dim, H, W)
        # h_prev: (B, hidden_dim, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        #i->inpute gate ; f->forget gate ; o->output gate ; g->candidate memory gate
        # The standard ConvLSTM update
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        #sigmoid->output values b/w 0 and 1 #0-> block; 1->open fully 

        c = f * c_prev + i * g #deep long term memory
        #forget old memory ; f-1-->keep memory #add new memory ; i-1-->add new memory
        #new cell state = kept old state + selected new memory 
        h = o * torch.tanh(c) #visible short memory
        #output gate --> mempry we expose
        return h, c


# XLSTM Encoder-Decoder

class XLSTM_Multi(nn.Module): #can now study both spatial structure
    def __init__(self, in_chan=1, hidden1=64, hidden2=96, future_steps=5):
        #in_chan = no.of channels in image frame 1-->grey scale ; 3->RGB
        #hidden1 = channels in first convolutional layer
        #hidden2 = channels in second convolutional layer

        super().__init__()
        self.future_steps = future_steps
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        # Encoder (Standard structure) --> takes input sequences compress all important information into a hidden representation
        self.enc1 = ConvLSTMCell(in_chan, hidden1)  #(B,1,H,W)-->enc1-->(B,64,H,W)
        self.enc2 = ConvLSTMCell(hidden1, hidden2)  #(B,64,H,W)-->enc2-->(B,96,H,W)

        # Decoder (Takes the previous frame/prediction as input, which is in_chan)
        self.dec1 = ConvLSTMCell(in_chan, hidden1)
        self.dec2 = ConvLSTMCell(hidden1, hidden2)

        self.conv_out = nn.Conv2d(hidden2, in_chan, kernel_size=1)  #(B,96,H,W)-->(B,1,H,W)

    def forward(self, inp_seq, future=None):
        # inp_seq: (B, T, C, H, W) 
        if future is None:
            future = self.future_steps

        B, T, C, H, W = inp_seq.shape
        device = inp_seq.device

        if T == 0:
            raise ValueError("Input sequence length T is zero. Check dataset / loader.")

        # init encoder states
        # The original code used h1, c1, h2, c2 for encoder states. We will reuse them.
        h1 = torch.zeros(B, self.hidden1, H, W, device=device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, self.hidden2, H, W, device=device)
        c2 = torch.zeros_like(h2)

        # Encoder pass to build context
        for t in range(T):
            x = inp_seq[:, t]          # (B, C, H, W)
            h1, c1 = self.enc1(x, h1, c1)
            h2, c2 = self.enc2(h1, h2, c2)
     
        # This transfers the context learned from the input sequence.        
        # The decoder hidden state (dh2, dc2) should match the last encoder state (h2, c2) and similarly for the first layer (dh1, dc1) should match (h1, c1).
        
        dh1, dc1 = h1.clone(), c1.clone()
        dh2, dc2 = h2.clone(), c2.clone()

        # Decoder input starts with the last frame of the encoder sequence
        dec_in = inp_seq[:, -1]

        preds = []
        for step in range(future):
            # Pass the previous layer's hidden state (h_prev) as input to the next layer but *only* the first ConvLSTM layer receives the image/prediction (dec_in).
            # The architecture is:
            # dec1(dec_in) -> dh1, dc1
            # dec2(dh1)    -> dh2, dc2

            dh1, dc1 = self.dec1(dec_in, dh1, dc1)
            dh2, dc2 = self.dec2(dh1, dh2, dc2)

            # Output is derived from the final hidden state of the decoder stack (dh2)
            out = torch.sigmoid(self.conv_out(dh2))
            preds.append(out)

            #The current prediction becomes the input for the next step
            dec_in = out

        return torch.stack(preds, dim=1)  # (B, future, C, H, W)


# Dataset 

class SeqDataset(Dataset):
    def __init__(self, folder, past=10, future=5, size=128, allow_small=False):
        self.past = past
        self.future = future
        self.size = size
        self.allow_small = allow_small
         
        #opens,sorts images with png in the folder
        files = sorted(glob(os.path.join(folder, "*.png")))
        frames = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Warning] could not load image: {f}")
                continue
            img = cv2.resize(img, (size, size))
            frames.append(img.astype(np.float32) / 255.0)

        self.frames = np.array(frames)  # (N, H, W)
        if self.frames.size == 0:
            print("[Warning] No frames found in dataset folder:", folder)

        # optional tiny jitter to avoid exact-constant images
        if self.frames.size != 0:
            # Reduced jitter magnitude to 1e-7 to be safer
            self.frames = self.frames + 1e-7 * np.random.randn(*self.frames.shape)
            # clip to [0,1]
            self.frames = np.clip(self.frames, 0.0, 1.0)

    def __len__(self):
        n = len(self.frames) - (self.past + self.future) + 1
        return max(0, n)

    def __getitem__(self, idx):
        # safe indexing
        if idx < 0:
            idx = 0
        if idx >= len(self):
            raise IndexError("Index out of range in SeqDataset")

        start = idx
        seq_in = self.frames[start: start + self.past]              # (past, H, W)
        seq_out = self.frames[start + self.past: start + self.past + self.future]  # (future, H, W)

        # final safety
        if seq_in.shape[0] != self.past or seq_out.shape[0] != self.future:
            # Using a more robust check that handles the case where the dataset is too small
            raise ValueError(f"Insufficient frames for idx {idx}: got {seq_in.shape[0]} in / {seq_out.shape[0]} out")

        # convert to tensors: (past, 1, H, W) and (future, 1, H, W)
        seq_in = torch.tensor(seq_in).unsqueeze(1).float()
        seq_out = torch.tensor(seq_out).unsqueeze(1).float()
        return seq_in, seq_out



# Loss helpers (works with (B,T,C,H,W) tensors)

def gradient_loss(pred, target):
    """
    Compute gradient difference loss for 5D tensors:
    pred, target shape = (B, T, C, H, W)
    We reshape to (-1, C, H, W) and compute simple absolute gradient differences.
    """
    # flatten time+batch
    p = pred.reshape(-1, pred.shape[2], pred.shape[3], pred.shape[4])  # (B*T, C, H, W)
    t = target.reshape(-1, target.shape[2], target.shape[3], target.shape[4])

    pdx = p[:, :, :, 1:] - p[:, :, :, :-1]
    tdx = t[:, :, :, 1:] - t[:, :, :, :-1] #-->horizantal edges
    pdy = p[:, :, 1:, :] - p[:, :, :-1, :]
    tdy = t[:, :, 1:, :] - t[:, :, :-1, :] #-->vertical edges

    # Use L2 (squared) gradient loss for stability, which is common in literature and often improves sharpness compared to L1 (absolute).
    return (pdx - tdx).pow(2).mean() + (pdy - tdy).pow(2).mean()
#-->gradient loss why? -->make predicted frames sharp by matching thier edges to the real edges


# Training loop 

def train_one_epoch(model, loader, opt, device, use_gdl=True, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        seq_in, seq_out = batch  # seq_in: (B, past, 1, H, W)
        if seq_in.shape[1] == 0:
            continue

        seq_in = seq_in.to(device)
        seq_out = seq_out.to(device)

        opt.zero_grad() #prevents any old gradients from accumulating
        preds = model(seq_in, future=seq_out.shape[1])  # (B, future, 1, H, W)

        # ensure shapes match
        if preds.shape != seq_out.shape:
            # safety: if mismatch, try to align by truncation/padding (rare)
            min_t = min(preds.shape[1], seq_out.shape[1])
            preds = preds[:, :min_t]
            seq_out = seq_out[:, :min_t]

        mse = F.mse_loss(preds, seq_out)
        if use_gdl:
            try:
                # Increased GDL weight to 1.0 to encourage sharper predictions
                # (blurry output often requires more emphasis on gradients)
                gdl = gradient_loss(preds, seq_out)
                loss = mse + 1.0 * gdl
            except Exception as e:
                # If gradient loss calculation fails, fallback to mse
                # print("[Warning] gradient_loss failed:", e) # Commented out for cleaner output
                loss = mse
        else:
            loss = mse

        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()

        total_loss += loss.item()
        n_batches += 1

    return (total_loss / n_batches) if n_batches > 0 else float('nan')



# MAIN 

def main():
    
    DATA_FOLDER = "/content/drive/MyDrive/OELP_XLSTM/final/rdataset"   # <- CHANGE THIS to your dataset path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH = 4
    EPOCHS = 100
    PAST = 10
    FUTURE = 5
    SIZE = 128
    LR = 1e-4

    print("Device:", DEVICE)
    dataset = SeqDataset(DATA_FOLDER, past=PAST, future=FUTURE, size=SIZE)
    if len(dataset) == 0:
        print("[ERROR] Dataset is empty. Check DATA_FOLDER and image files.")
        return

    print("Dataset size:", len(dataset), "sequences")
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=False)

    model = XLSTM_Multi(in_chan=1, hidden1=64, hidden2=96, future_steps=FUTURE).to(DEVICE)
    # Using AdamW instead of Adam is a common practice for better generalization
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # training
    print("Training started...")
    for epoch in range(EPOCHS):
        # Increased clip_grad to 5.0 (can help with large gradients)
        loss = train_one_epoch(model, loader, opt, DEVICE, use_gdl=True, clip_grad=5.0)
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss = {loss:.6f}")

        # save checkpoints occasionally
        if (epoch + 1) % 10 == 0:
            ckpt = f"xlstm_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt)
            print("Saved checkpoint:", ckpt)

    torch.save(model.state_dict(), "xlstm_final.pth")
    print("Model saved as xlstm_final.pth")

    #INFERENCE (checks what the model actually learned) 
    try:
        # Use a random sample for validation to ensure model is generalizing, not just memorizing the last sequence
        idx_inf = random.randint(0, len(dataset) - 1)
        seq_in, seq_out = dataset[idx_inf]
        print(f"Inference run on random sequence index: {idx_inf}")
    except Exception as e:
        print("[Error] Could not fetch a sequence for inference:", e)
        return

    seq_in = seq_in.unsqueeze(0).to(DEVICE)   # (1, past, 1, H, W)
    seq_out = seq_out.unsqueeze(0).to(DEVICE) # (1, future, 1, H, W)
    model.eval()
    with torch.no_grad():
        preds = model(seq_in, future=seq_out.shape[1]).detach().cpu()  # (1, future, 1, H, W)
        seq_out_cpu = seq_out.detach().cpu()

    # save predicted frames to disk
    os.makedirs("pred_output", exist_ok=True)
    for i in range(preds.shape[1]):
        # Ensure output is a proper image (denormalized, single channel)
        img = (preds[0, i, 0].numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"pred_output/pred_{i}.png", img)

    # also save actual future frames from seq_out
    for i in range(seq_out_cpu.shape[1]):
        img = (seq_out_cpu[0, i, 0].numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"pred_output/actual_{i}.png", img)

    print("Predictions saved in pred_output/")

    # optional: create a simple comparison plot (actual vs pred images)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, preds.shape[1], figsize=(3*preds.shape[1], 6))
        for i in range(preds.shape[1]):
            axes[0, i].imshow(seq_out_cpu[0, i, 0], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Actual t+{i+1}")

            axes[1, i].imshow(preds[0, i, 0], cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Pred t+{i+1}")

        plt.tight_layout()
        plt.savefig("pred_vs_actual.png", dpi=200)
        print("Saved visual comparison as pred_vs_actual.png")
    except Exception as e:
        print("[Warning] Could not save comparison plot:", e)


if __name__ == "__main__":
    main()
