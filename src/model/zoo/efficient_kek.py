import geffnet
import torch

from torch import nn, Tensor

from .samwise import RNNBlock


def build_encoder(name: str, weights: str, freeze=True) -> nn.Module:
    model = geffnet.create_model(name, pretrained=False)
    weights = torch.load(weights)
    model.load_state_dict(weights)
    model = nn.Sequential(*list(model.children())[:-1])
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    model.eval()
    return model


class EfficientKek(nn.Module):
    def __init__(self, bb: str, bb_weights: str, rnn_in: int,
                 rnn_out: int, p_out_drop: float):
        super(EfficientKek, self).__init__()
        self.encoder = build_encoder(bb, bb_weights, freeze=True)
        self.rnn = RNNBlock(rnn_in, rnn_out, bidirectional=True)

        out_in = self.rnn.out_ch
        out = [nn.Linear(out_in, out_in // 4),
               nn.ReLU(inplace=True),
               nn.Linear(out_in // 4, 1, bias=False)]
        if p_out_drop > 0:
            out = [nn.Dropout(p=p_out_drop)] + out
        self.out = nn.Sequential(*out)

    def forward(self, x: Tensor, y=None):
        N, C, D, H, W = x.shape
        seq = []

        for f in range(D):
            h = self.encoder(x[:, :, f])
            # N, C, 1, 1 -> N, C, D
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
        return y_pred.clamp_(0.05, 0.95)
