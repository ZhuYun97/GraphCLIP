from transformers import AutoModel
import numpy as np
import torch
from torch import Tensor
from peft import get_peft_model, LoraConfig, TaskType, LoraModel, PromptTuningConfig

from .projector import Projector
from .gt import GPS



#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



text_ids = {
    'tiny': 'sentence-transformers/all-MiniLM-L6-v2',
    'sbert':  'sentence-transformers/multi-qa-distilbert-cos-v1', #'sentence-transformers/all-MiniLM-L6-v2', #'sentence-transformers/multi-qa-distilbert-cos-v1',
    'e5': 'intfloat/e5-base-v2',
    'deberta': 'microsoft/deberta-v3-base',
}



class Config:
    def __init__(self):
        self.prefix_projection=True
        self.pre_seq_len=10
        self.num_hidden_layers=6
        self.prefix_hidden_size=384
        self.hidden_size=384

config = Config()

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class GraphCLIP(torch.nn.Module):
    def __init__(self, graph_input_dim, graph_hid_dim, graph_num_layer, attn_kwargs, text_model='tiny'):
        super().__init__()
        self.graph_model = GPS(in_dim=graph_input_dim, channels=graph_hid_dim, out_dim=graph_hid_dim, 
                               pe_dim=8, num_layers=graph_num_layer, attn_type='multihead', attn_kwargs=attn_kwargs)
        self.text_model_type = text_model
        text_id = text_ids[text_model]
        text_model = AutoModel.from_pretrained(text_id)
        self.text_model = text_model
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def encode_graph(self, batch):
        graph_embs, center_embs = self.graph_model(batch.x, batch.pe, batch.edge_index, batch.batch, batch.root_n_index)
        return graph_embs, center_embs

    def encode_text(self, input_ids, token_type_ids, attention_mask):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embs = mean_pooling(text_output.last_hidden_state, attention_mask)
        return text_embs

    def forward(self, batch_g, batch_t):
        graph_features, c_features = self.encode_graph(batch_g)
        text_features = self.encode_text(**batch_t)

        # normalized features
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_graph = logit_scale * graph_features @ text_features.t()
        logits_per_text = logits_per_graph.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_graph, logits_per_text
    
    def freeze_text(self):
        for k, v in self.text_model.named_parameters():
            v.requires_grad = False

    
    # used for prompt tuning
    def sup_loss(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    