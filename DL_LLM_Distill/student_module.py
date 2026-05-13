import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DataConfig, ModelConfig, TrainConfig
from dataset_module import dataset_module

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def student_module(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, training=False):
    df_train = data_cfg.df_train
    df_val = data_cfg.df_val
    cols = data_cfg.cols

    context = model_cfg.context
    horizon = model_cfg.horizon
    input_dim = model_cfg.input_dim

    train_preds = train_cfg.train_preds
    val_preds = train_cfg.val_preds

    # ----- Append teacher predictions to distillation dataset -----
    class AppendDataset(torch.utils.data.Dataset):
        def __init__(self, df, context, horizon, teacher_preds):
            self.X_context = []
            self.X_future = []
            self.Y = []
            self.teacher_preds = teacher_preds
            periods = context + horizon
            for i in range(len(df) - periods + 1):
                window = df.iloc[i:i+periods]
                expect = pd.date_range(start=window['TmStamp'].iloc[0], periods=periods, freq='h')
                if not window['TmStamp'].reset_index(drop=True).equals(pd.Series(expect)):
                    continue
                values = window[['H'] + cols].values.astype(np.float32)
                self.X_context.append(values[:context])
                self.X_future.append(values[context:context+horizon, 1:])
                self.Y.append(values[context:context+horizon, 0])

        def __len__(self):
            return len(self.X_context)

        def __getitem__(self, idx):
            return (torch.tensor(self.X_context[idx], dtype=torch.float32),
                    torch.tensor(self.X_future[idx], dtype=torch.float32),
                    torch.tensor(self.Y[idx], dtype=torch.float32),
                    torch.tensor(self.teacher_preds[idx], dtype=torch.float32))

    # ----- Student (TCN encoder + decoder) architecture -----
    class ResidualBlock(nn.Module):
        def __init__(self, channels, dilation, kernel_size=3):
            super().__init__()
            padding = (kernel_size - 1) * dilation // 2
            # Dilated Conv1d with shape (B, C, T)
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
            # LayerNorm with shape (B, T, C)
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)

        def forward(self, x):
            residual = x
            # First Conv1d + LayerNorm + Activation
            # (B, C, T) -> (B, T, C) for LayerNorm
            out = self.conv1(x).transpose(1, 2)
            out = self.norm1(out)
            out = F.gelu(out).transpose(1, 2)  # Back to (B, C, T)
            # Second Conv1d + LayerNorm + Activation
            out = self.conv2(out).transpose(1, 2)
            out = self.norm2(out)
            out = F.gelu(out).transpose(1, 2)  # Back to (B, C, T)
            # Residual connection
            return out + residual

    class TCNEncoder(nn.Module):
        def __init__(self, input_dim=input_dim, hidden_dim=64):
            super().__init__()
            self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            # Dilations expand receptive field exponentially
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]
            self.encoder_blocks = nn.ModuleList([ResidualBlock(hidden_dim, d) for d in dilations])
            # Compress temporal dimension to fixed length
            self.pool = nn.AdaptiveAvgPool1d(1)
        
        def forward(self, x):
            # (B, T, C) -> (B, C, T) for Conv1d
            x = self.input_proj(x.transpose(1, 2))
            # Pass through stacked dilated residual blocks
            for block in self.encoder_blocks:
                x = block(x)
            # Temporal compression to (B, hidden, 1)
            x = self.pool(x).squeeze(-1)  # (B, hidden)
            return x

    class TCNDecoder(nn.Module):
        def __init__(self, horizon=horizon, future_dim=len(cols), hidden_dim=64):
            super().__init__()
            self.horizon = horizon
            self.future_proj = nn.Linear(future_dim, hidden_dim)
            # Dilations expand receptive field exponentially
            dilations = [1, 2, 4, 8, 16, 32, 64, 128]
            self.decoder_blocks = nn.ModuleList([ResidualBlock(hidden_dim*2, d) for d in dilations])
            self.output_proj = nn.Conv1d(hidden_dim*2, 1, kernel_size=1)
        
        def forward(self, latent, x_future):
            h_context = latent.unsqueeze(1)                              # (B, 1, hidden)
            h_context = h_context.repeat(1, self.horizon, 1)             # (B, 12, hidden)
            h_future = self.future_proj(x_future)                        # (B, 12, hidden)
            h = torch.cat([h_context, h_future], dim=-1).transpose(1,2)  # (B, hidden*2, 12)
            for block in self.decoder_blocks:
                h = block(h)
            return self.output_proj(h).squeeze(1)                        # (B, 12)

    class StudentModel(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x_context, x_future):
            latent = self.encoder(x_context)
            pred = self.decoder(latent, x_future)
            return pred

    # ----- Student training -----
    epochs, batch_size = 100, 64
    student = StudentModel(encoder=TCNEncoder(), decoder=TCNDecoder()).cuda()
    optimizer_student = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    
    if training:
        train_ds = AppendDataset(df_train, context, horizon, train_preds)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_ds = AppendDataset(df_val, context, horizon, val_preds)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        # Loss function
        mse_loss = nn.MSELoss()
        def distill_loss(y_pred, teacher_pred, y_true):
            return 0.7 * mse_loss(y_pred, teacher_pred) + 0.3 * mse_loss(y_pred, y_true)

        # Training loop
        best_val_loss, patience, counter = float('inf'), 10, 0
        for epoch in range(epochs):
            # Training
            student.train()
            train_loss = 0
            for x_context, x_future, y_true, teacher_pred in train_loader:
                x_context = x_context.cuda(non_blocking=True)
                x_future = x_future.cuda(non_blocking=True)
                y_true = y_true.cuda(non_blocking=True)
                teacher_pred = teacher_pred.cuda(non_blocking=True)
                optimizer_student.zero_grad()
                y_pred = student(x_context, x_future)
                loss = distill_loss(y_pred, teacher_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer_student.step()
                train_loss += loss.item()
            # Validation
            student.eval()
            val_loss = 0
            with torch.no_grad():
                for x_context, x_future, y_true, teacher_pred in val_loader:
                    x_context = x_context.cuda(non_blocking=True)
                    x_future = x_future.cuda(non_blocking=True)
                    y_true = y_true.cuda(non_blocking=True)
                    teacher_pred = teacher_pred.cuda(non_blocking=True)
                    y_pred = student(x_context, x_future)
                    loss = distill_loss(y_pred, teacher_pred, y_true)
                    val_loss += loss.item()
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer_student.state_dict(),
                    'val_loss': val_loss
                }, 'best_student.pt')
            else:
                counter += 1
                if counter >= patience:
                    break
            print(f'Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')
    
    return student, optimizer_student

def main():
    # Dataset module
    cols, _, df_train, df_val = dataset_module()
    data_cfg = DataConfig(df_train=df_train, df_val=df_val, cols=cols)
    model_cfg = ModelConfig(context=1024, horizon=12, input_dim=len(cols)+1)
    # Teacher module
    train_preds = np.load('train_preds_p50.npy')
    val_preds = np.load('val_preds_p50.npy')
    train_cfg = TrainConfig(train_preds, val_preds)
    # Student module
    student_module(data_cfg, model_cfg, train_cfg, training=True)

if __name__ == "__main__":
    main()