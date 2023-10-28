import torch
import transformers as tfm
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
from .util import get_extended_attention_mask, pad_mask
from .util import get_ext_local_attention_mask


class GlobalCorrector(torch.nn.Module):
    def __init__(self, layers, char_model, head_model):
        super().__init__()
        self.char_model = char_model
        conf = tfm.BertConfig(vocab_size=1, num_hidden_layers=layers)
        self.embeddings = BertEmbeddings(conf)
        self.encoder = BertEncoder(conf)
        self.head = head_model
        # self.embeddings.apply(_init_weights)
        # self.encoder.apply(_init_weights)

    def forward(self, seqs_of_tokens, seqs_of_choices, seqs_of_labels=None):
        in_emb = self.embeddings(inputs_embeds=self.char_model(seqs_of_tokens))
        mask = pad_mask(seqs_of_tokens, pad=2, batch_first=True).to(in_emb.device)
        hid = self.encoder(in_emb, attention_mask=get_extended_attention_mask(mask, in_emb.dtype)).last_hidden_state

        return self.head(hid[:, 1:-1], seqs_of_choices, seqs_of_labels)


# class FiRo(torch.nn.Module):
#     def __init__(self, char_model, head_model):
#         super().__init__()
#         self.char_model = char_model
#         conf = tfm.BertConfig(vocab_size=1)
#         self.embeddings = BertEmbeddings(conf)
#         self.head = head_model
#         self.coef = torch.nn.Parameter(torch.zeros(1))

#     def forward(self, seqs_of_tokens, seqs_of_choices, seqs_of_labels=None):
#         in_emb = self.embeddings(inputs_embeds=self.char_model(seqs_of_tokens))
#         alpha = torch.sigmoid(self.coef)
#         beta = (1 - alpha) / 2
#         avg = alpha*in_emb[:, 1:-1] + beta*in_emb[:, 2:] + beta*in_emb[:, :-2]
#         return self.head(avg, seqs_of_choices, seqs_of_labels)


class FiRo(torch.nn.Module):
    def __init__(self, layers, char_model, head_model, attention_size):
        super().__init__()
        self.char_model = char_model
        conf = tfm.BertConfig(vocab_size=1, num_hidden_layers=layers)
        self.embeddings = BertEmbeddings(conf)
        self.encoder = BertEncoder(conf)
        self.head = head_model
        self.attention_size = attention_size

    def forward(self, seqs_of_tokens, seqs_of_choices, seqs_of_labels=None):
        in_emb = self.embeddings(inputs_embeds=self.char_model(seqs_of_tokens))
        mask = pad_mask(seqs_of_tokens, pad=2, batch_first=True).to(in_emb.device)
        attention_mask = get_ext_local_attention_mask(mask, self.attention_size, in_emb.dtype)
        hid = self.encoder(in_emb, attention_mask=attention_mask).last_hidden_state

        return self.head(hid[:, 1:-1], seqs_of_choices, seqs_of_labels)
