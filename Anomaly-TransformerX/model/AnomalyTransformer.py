import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embedding import DataEmbedding, TokenEmbedding

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=100):

        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
        self.projection=nn.Linear(256,38)

    def forward(self, x):

        residual = x  # （1024*100*256）
        out = F.relu(self.norm1(self.conv1(x)))  #  1024*100*38


        out = self.norm2(self.conv2(out))  #  1024*100*38

        # （256→38）
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)  # 1024*100*256 → 1024*100*38
            
        final=out + residual 
        final=self.projection(final) # L:256->38
        return final  # 1024*100*38
 
class EncoderLayer(nn.Module):
    def __init__(self, 
                 attention, 
                 d_model, 
                 d_ff=None, 
                 dropout=0.1, 
                 activation="relu"):
        
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn
class Encoder(nn.Module):
    def __init__(self, 
                 attn_layers, 
                 norm_layer=None):
    
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []

        
        for attn_layer in self.attn_layers:
            x, series = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list

class CustomLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=38, num_layers=1, batch_first=True):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)  
        out = self.dropout(out) 
        out = self.fc(out)
        return out
    
class AnomalyTransformer(nn.Module):
    def __init__(
                 self, 
                 win_size, 
                 enc_in, 
                 c_out, 
                 alpha,
                 beta,
                 d_model=256, 
                 n_heads=8, 
                 e_layers=3, 
                 d_ff=512,
                 dropout=0.05, 
                 activation='gelu', 
                 output_attention=True
                 ):
        
        super(AnomalyTransformer, self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout) # (B,W,d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.resnet = ResidualBlock(in_channels=100)
        self.lstm = CustomLSTM()


    def forward(self, x):
        
        enc_out = self.embedding(x) # [1024, 100, 256]
        lstm_out=self.lstm(enc_out)
        res_out = self.resnet(enc_out) 
        enc_out, series = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        final_out=enc_out+self.alpha*lstm_out+self.beta*res_out

        if self.output_attention:
            return final_out, series
        else:
            return final_out  # [B, L, D]
