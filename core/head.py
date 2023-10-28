import torch
import torch.nn.functional as F
from .util import _init_weights, pad_mask


class VarLinear(torch.nn.Module):
    def __init__(self, d_model):
        super(VarLinear, self).__init__()
        self.e_weight = torch.nn.Embedding(100_010, d_model, padding_idx=0)
        _init_weights(self.e_weight)

    def _pred(self, last_hid, seq_of_choices):
        dev = next(self.parameters()).device

        pidx = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(t) for t in seq_of_choices], batch_first=True).to(dev)
        weight = self.e_weight(pidx)

        valid = pad_mask(seq_of_choices, batch_first=True).to(dev) == 1
        pred = torch.matmul(weight, last_hid[:len(seq_of_choices), :, None]).squeeze(2)
        logits = torch.where(valid, pred, torch.full_like(pred, -float('inf')))

        assert pidx.shape == pred.shape == logits.shape, (pidx.shape, pred.shape, logits.shape)

        y_pred = pidx.gather(dim=1, index=logits.max(1, keepdims=True)[1]).squeeze(1)

        return y_pred, torch.log_softmax(logits, dim=-1)

    def forward(self, hid, seqs_of_choices, tag_seqs=None):
        dev = next(self.parameters()).device
        loss = lengths = 0
        out = []
        for i, seq_of_choices in enumerate(seqs_of_choices):
            y_pred, logits = self._pred(hid[i], seq_of_choices)
            out.append(y_pred)
            if tag_seqs is not None:
                loss += F.nll_loss(logits, torch.tensor(tag_seqs[i], device=dev), reduction='sum')
            lengths += len(seq_of_choices)

        loss /= lengths
        return loss, out
