import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from allennlp.nn.util import batched_index_select
from transformers import BertModel, AutoModelForMaskedLM
from typing import Optional


class DualBert(nn.Module):

    def __init__(self,
                 pretrained_model_name,
                 use_same_model=False,
                 idiom_mask_length=4,
                 mode='cosine_similarity',
                 linear_hidden_size=256,
                 use_cls=True,
                 use_generation=True):
        """
        Args:
            pretrained_model_name: name of pretrained model
            use_same_model: whether to use the same model for sentence and idiom
            idiom_mask_length: length of idiom mask
            mode: 'cosine_similarity','euclidean_distance','linear','cross_attention'
            linear_hidden_size: hidden size of linear layer in linear mode
            use_cls: whether to use [CLS] token in idiom pattern or use idiom tokens
            use_generation: whether to use generation head to predict idiom
        """

        super(DualBert, self).__init__()
        self.bert_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name)

        self.sentence_model = self.bert_model.bert
        if use_same_model:
            self.idiom_model = self.sentence_model
        else:
            self.idiom_model = BertModel.from_pretrained(pretrained_model_name)

        self.sentence_pooler = nn.Sequential(
            nn.Linear(self.sentence_model.config.hidden_size * idiom_mask_length,
                      self.sentence_model.config.hidden_size),
            nn.Tanh()
        )

        self.use_cls = use_cls
        if not use_cls:
            self.idiom_pooler = nn.Sequential(
                nn.Linear(self.idiom_model.config.hidden_size * idiom_mask_length,
                          self.idiom_model.config.hidden_size),
                nn.Tanh()
            )

        self.use_generation = use_generation
        if use_generation:
            self.gen_cls = self.bert_model.cls

        self.mode = mode
        if self.mode == 'linear':
            self.aggregation = nn.Sequential(
                nn.Linear(self.sentence_model.config.hidden_size + self.idiom_model.config.hidden_size,
                          linear_hidden_size),
                nn.Tanh(),
                nn.Linear(linear_hidden_size, 1)
            )
        elif self.mode == 'cross_attention':
            self.cross_attention = nn.MultiheadAttention(self.sentence_model.config.hidden_size, 8)
            self.aggregation = nn.Sequential(nn.Tanh(), nn.Linear(self.sentence_model.config.hidden_size, 1))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.idiom_mask_length = idiom_mask_length
        self.use_same_model = use_same_model

    def new_params(self):
        if self.mode == 'linear':
            params = [self.sentence_pooler, self.aggregation, self.logit_scale]
        elif self.mode == 'cross_attention':
            params = [self.sentence_pooler, self.cross_attention, self.aggregation, self.logit_scale]
        else:
            params = [self.sentence_pooler, self.logit_scale]

        if not self.use_cls:
            params.append(self.idiom_pooler)

        return params

    def pretrained_params(self):
        return [self.sentence_model] if self.use_same_model \
            else [self.sentence_model, self.idiom_model]

    def embed_sentence(self, input_ids, attention_mask, candidate_mask, candidate_index=None):
        """
        Args:
            input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
            attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
            candidate_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
            candidate_index: torch.LongTensor of shape `(batch_size, candidate_num, idiom_mask_length)`
        """

        # (batch_size, sequence_length, hidden_size)
        sentence_outputs = self.sentence_model(input_ids, attention_mask=attention_mask)[0]
        # (batch_size, idiom_mask_length*hidden_size)
        idiom_outputs = torch.masked_select(sentence_outputs, candidate_mask.unsqueeze(-1)).reshape(
            -1, self.idiom_mask_length * sentence_outputs.shape[-1])

        if self.use_generation:
            # (batch_size * idiom_mask_length, vocab_size)
            idiom_pred = self.gen_cls(idiom_outputs.reshape(-1, sentence_outputs.shape[-1])).unsqueeze(-1)

            # (batch_size * idiom_mask_length, candidate_num)
            candidate_index = candidate_index.transpose(-1, -2)
            candidate_index = candidate_index.reshape(-1, candidate_index.shape[-1])

            # (batch_size * idiom_mask_length, candidate_num)
            idiom_logits = batched_index_select(idiom_pred, candidate_index).squeeze(-1)
            # (batch_size, candidate_num, idiom_mask_length)
            idiom_logits = idiom_logits.reshape(-1, self.idiom_mask_length, idiom_logits.shape[-1]).transpose(-1, -2)
        else:
            idiom_logits = None

        # (batch_size, hidden_size)
        idiom_outputs = self.sentence_pooler(idiom_outputs)
        idiom_outputs = idiom_outputs / torch.norm(idiom_outputs, dim=-1, keepdim=True)
        if self.use_generation:
            return idiom_outputs, idiom_logits
        else:
            return idiom_outputs

    def embed_idiom_pattern(self, input_ids, attention_mask):
        """
        Args:
            input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
            attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        """
        # (batch_size, hidden_size)

        if self.use_cls:
            idiom_pattern_outputs = self.idiom_model(input_ids, attention_mask=attention_mask)[1]
            idiom_pattern_outputs = idiom_pattern_outputs / torch.norm(idiom_pattern_outputs, dim=1, keepdim=True)
        else:
            # (batch_size, sequence_length, hidden_size)
            idiom_pattern_outputs = self.idiom_model(input_ids, attention_mask=attention_mask)[0]
            # (batch_size, idiom_mask_length*hidden_size)
            idiom_pattern_outputs = idiom_pattern_outputs[:, 1:1 + self.idiom_mask_length, :].reshape(
                -1, self.idiom_mask_length * idiom_pattern_outputs.shape[-1])
            # (batch_size, hidden_size)
            idiom_pattern_outputs = self.idiom_pooler(idiom_pattern_outputs)
            idiom_pattern_outputs = idiom_pattern_outputs / torch.norm(idiom_pattern_outputs, dim=1, keepdim=True)

        return idiom_pattern_outputs

    def calculate_logits(self, sentence_outputs, idiom_pattern_outputs):
        """
        Args:
            sentence_outputs: torch.FloatTensor of shape `(batch_size, hidden_size)`
            idiom_pattern_outputs: torch.FloatTensor of shape `(batch_size, candidate_num, hidden_size)`
        """
        if self.mode == 'cosine_similarity':
            logits = torch.matmul(sentence_outputs.unsqueeze(-2), idiom_pattern_outputs.transpose(1, 2)).squeeze(-2)
        elif self.mode == 'euclidean_distance':
            logits = -torch.norm(sentence_outputs.unsqueeze(-2) - idiom_pattern_outputs, dim=-1)
        elif self.mode == 'linear':
            candidate_num = idiom_pattern_outputs.shape[1]
            logits = self.aggregation(torch.cat([sentence_outputs.unsqueeze(-2).repeat(1, candidate_num, 1),
                                                 idiom_pattern_outputs],
                                                dim=-1)).squeeze(-1)
        elif self.mode == 'cross_attention':
            logits = self.cross_attention(idiom_pattern_outputs.transpose(0, 1),
                                          sentence_outputs.unsqueeze(0),
                                          sentence_outputs.unsqueeze(0))[0]  # (candidate_num, batch_size, hidden_size)
            logits = self.aggregation(logits.transpose(0, 1)).squeeze(-1)  # (batch_size, candidate_num)
        else:
            raise ValueError('mode must be cosine_similarity or euclidean_distance or linear or cross_attention')

        return logits

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            candidate_mask: torch.Tensor,
            candidate_pattern: torch.Tensor,
            candidate_pattern_mask: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
    ):
        r"""
        input_ids: torch.LongTensor of shape `(batch_size, sequence_length)`
        attention_mask: torch.LongTensor of shape `(batch_size, sequence_length)`
        candidate_mask: torch.LongTensor of shape `(batch_size, sequence_length)`

        candidate_pattern: torch.LongTensor of shape `(batch_size, candidate_num, idiom_pattern_length)`
        candidate_pattern_mask: torch.LongTensor of shape `(batch_size, candidate_num, idiom_pattern_length)`

        labels: torch.LongTensor of shape `(batch_size, )`
        """
        with torch.no_grad():
            self.logit_scale.data.clamp_(-100, 100)
        if self.use_generation:
            candidate_index = candidate_pattern[:, :, 1:1 + self.idiom_mask_length]
            # (batch_size, hidden_size), (batch_size, candidate_num, idiom_mask_length)
            idiom_outputs, idiom_logits = self.embed_sentence(input_ids, attention_mask, candidate_mask,
                                                              candidate_index)
        else:
            # (batch_size, hidden_size)
            idiom_outputs = self.embed_sentence(input_ids, attention_mask, candidate_mask)
            idiom_logits = None

        B, N, L = candidate_pattern.shape
        candidate_pattern = candidate_pattern.reshape(-1, L)
        candidate_pattern_mask = candidate_pattern_mask.reshape(-1, L)
        # (batch_size*candidate_num, hidden_size)
        idiom_pattern_outputs = self.embed_idiom_pattern(candidate_pattern, candidate_pattern_mask)

        # (batch_size, candidate_num, hidden_size)
        candidate_pattern_outputs = idiom_pattern_outputs.reshape(B, N, -1)
        # (batch_size, candidate_num)
        logits = self.calculate_logits(idiom_outputs, candidate_pattern_outputs)
        logits *= self.logit_scale

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if self.use_generation:
                loss += F.cross_entropy(idiom_logits, labels.unsqueeze(-1).repeat(1, self.idiom_mask_length))
            return loss
        else:
            pt = F.softmax(logits, dim=-1)
            if self.use_generation:
                idiom_pt = F.softmax(idiom_logits, dim=-2)
                pt1 = pt + idiom_pt.sum(-1)
                pt2 = pt + idiom_pt.mean(-1)
                pt3 = self.idiom_mask_length * pt.log() + idiom_pt.log().sum(-1)
                return torch.stack((pt1.argmax(-1), pt2.argmax(-1), pt3.argmax(-1)), dim=-1)
            else:
                return pt.argmax(dim=-1)
