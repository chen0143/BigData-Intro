import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig(max_position_embeddings=512), cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self.size_embedding = size_embedding
        # layers
        # self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types)
        self.rel_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3 + size_embedding * 2 ,(config.hidden_size * 3 + size_embedding * 2)//2),
            nn.PReLU(),
            #inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出利用in-place计算可以节省内（显）存
            # nn.Dropout(p=0.1),#默认值：0.5
            nn.Linear((config.hidden_size * 3 + size_embedding * 2)//2,  relation_types)    
        )
        # self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.width_embeddings = nn.Embedding(500, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        # weight initialization
        self.init_weights()

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor,
                        rel_masks: torch.tensor, entity_types: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        entity_ctx = get_token(h, encodings, self._cls_token)
        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, 
        h, entity_masks, size_embeddings, entity_types)

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            "cuda")#"cuda"self.rel_classifier.weight.device

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, 
                           entity_sample_masks: torch.tensor, entity_types: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        entity_ctx = get_token(h, encodings, self._cls_token)
        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings,entity_types)
        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            "cuda" )#self.rel_classifier[0].weight.device

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings, entity_types):
        # max pool entity candidate spans
        # print(entity_masks.shape,entity_masks[0,0,:,])
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool2 = entity_spans_pool[:,:,0,:]
        shape1 = entity_spans_pool.shape
        for i in range(shape1[0]):
            for j in range(shape1[1]):
                for k in range(shape1[2]):
                    if entity_masks[i,j,k]:
                        entity_spans_pool2[i,j,:] = entity_spans_pool[i,j,k,:]
        entity_spans_pool = entity_spans_pool2
        # entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        entity_clf = entity_spans_pool.max(dim=2)[0].unsqueeze(-1).repeat(1,1,3)
        # get cls token as candidate context representation
        shape = entity_clf.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if k == entity_types[i,j]:
                        entity_clf[i, j, k] = 100.0
                    else:
                        entity_clf[i, j, k] = 0.0
        # create candidate representations including context, max pooled span and size embedding
        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]
        
        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)
        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)
        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0
        # rel_len = rel_masks.to(torch.uint8).sum(dim=-1).unsqueeze(-1)
        # rel_shape = rel_len.shape
        # rel_lens = rel_len.repeat(1,1,self.size_embedding)
        # for i in range(rel_shape[0]):
        #     for j in range(rel_shape[1]):
        #         rel_lens[i,j] = self.width_embeddings(rel_len[i,j])
        # rel_ctx = torch.cat([rel_lens, rel_ctx],dim=-1)
        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        # entity_ctx = entity_ctx.unsqueeze(1).repeat(1,rel_shape[1],1)
        rel_repr = torch.cat([rel_ctx, size_pair_embeddings, entity_pairs], dim=2)
        rel_repr = self.dropout(rel_repr)
        # classify relation candidates
        # chunk_rel_logits = self.rel_classifier(rel_repr)
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []
            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()
            # create relations and masks
            for s1,i1,  in enumerate(entity_logits_max[i]):
                for s2, i2 in enumerate(entity_logits_max[i]):
                    if i1 ==1 and i2==2:
                        rels.append((s1, s2))
                        rel_masks.append(sampling.create_rel_mask(entity_spans[i][s1], 
                                                        entity_spans[i][s2], ctx_size))
                        sample_masks.append(1)
            # for i1, s1 in zip(non_zero_indices, non_zero_spans):
            #     for i2, s2 in zip(non_zero_indices, non_zero_spans):
            #         if i1 != i2:
            #             rels.append((i1, i2))
            #             rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
            #             sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier[0].weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]
