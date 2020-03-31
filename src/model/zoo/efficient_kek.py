import geffnet
import torch

from ..fastai import RNNDropout
from torch import nn, Tensor
from typing import Optional, Tuple


class RNNBlock(nn.Module):
    def __init__(self, in_ch: int, rnn_ch: int, bidirectional=False,
                 p_embed=0.1, p_input=0.5):
        super().__init__()

        gru = nn.GRU(in_ch, rnn_ch, bidirectional=bidirectional, batch_first=True)
        layers = [gru]
        if p_input > 0:
            # set zeros in dim=1 (D)
            layers = [RNNDropout(p_input)] + layers
        if p_embed > 0:
            # last dim should be C
            layers = [nn.Dropout(p_embed, inplace=True)] + layers
        self.gru = nn.Sequential(*layers) if len(layers) > 1 else layers[0]

        self.out_ch = rnn_ch * 2 * (2 if bidirectional else 1)

    def forward(self, x):
        # N, C, D -> N, D, C
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x_mean = x.mean(1)
        x_max, _ = x.max(1)
        x = torch.cat([x_mean, x_max], dim=1)
        return x


def build_encoder(name: str, weights=None, freeze=True) -> Tuple[nn.Module, int]:
    model = geffnet.create_model(name, pretrained=False)
    if weights is not None:
        weights = torch.load(weights)
        model.load_state_dict(weights)
    layers = list(model.children())
    model = nn.Sequential(*layers[:-1])
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    x = torch.rand(1, 3, 224, 224)
    x = model(x)
    return model, x.shape[1]


class EfficientKek(nn.Module):
    def __init__(self, bb: str, rnn_out: int, bb_weights: Optional[str] = None,
                 p_embed=0.1, p_input=0.5, p_out=0.2):
        super(EfficientKek, self).__init__()
        self.encoder, enc_out_ch = build_encoder(bb, bb_weights)
        self.rnn = RNNBlock(enc_out_ch, rnn_out, bidirectional=True,
                            p_embed=p_embed, p_input=p_input)
        out_in = self.rnn.out_ch
        out = [nn.Linear(out_in, out_in // 4),
               nn.ReLU(inplace=True),
               nn.Linear(out_in // 4, 1, bias=False)]
        if p_out > 0:
            out = [nn.Dropout(p=p_out)] + out
        self.out = nn.Sequential(*out)

    def forward(self, x: Tensor, y=None):
        N, C, D, H, W = x.shape
        seq = []

        for f in range(D):
            h = self.encoder(x[:, :, f])
            h = h.squeeze(-1)
            seq.append(h)
        seq = torch.cat(seq, dim=2)
        seq_out = self.rnn(seq)
        out = self.out(seq_out)
        # for compatibility with other models
        return out, None

    @staticmethod
    def to_y(y_hat: Tensor, _: any):
        y_pred = y_hat.detach()
        y_pred = torch.sigmoid(y_pred).squeeze_(1)
        return y_pred.clamp_(0.1, 0.9)
