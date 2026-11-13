"""Two-head LSTM: one shared encoder, one head for attack DIAGNOSIS, one for fault LOCALIZATION."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TwoHeadLSTM(nn.Module):
    def __init__(self, vocab_size, n_diag, n_loc, n_cont=3, emb=16, hidden=64, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb + n_cont, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head_diag = nn.Linear(hidden, n_diag)
        self.head_loc = nn.Linear(hidden, n_loc)

    def forward(self, tok, cont):
        lengths = (tok != 0).sum(1).clamp(min=1).cpu()
        x = torch.cat([self.emb(tok), cont], dim=-1)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = self.drop(h[-1])
        return self.head_diag(h), self.head_loc(h)
