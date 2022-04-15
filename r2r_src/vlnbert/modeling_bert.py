# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

import sys

from pytorch_transformers.modeling_bert import (BertEmbeddings,
        BertSelfAttention, BertAttention, BertEncoder, BertLayer,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertPooler, BertLayerNorm, BertPreTrainedModel,
		BertPredictionHeadTransform)

logger = logging.getLogger(__name__)

class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else: # default
            mixed_query_layer = self.query(hidden_states) # [24, 95, 768]
            mixed_key_layer = self.key(hidden_states)     # [24, 95, 768]
            mixed_value_layer = self.value(hidden_states) # [24, 95, 768]

        # transpose into shape [24, 12, 95, 64] as [batch_size, num_heads, seq_length, feature_dim]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [24, 12, 95, 95]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs) # [24, 12, 95, 95]

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # [24, 12, 95, 64]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # [24, 95, 768]

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)

        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        ''' transformer processing '''
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)
        ''' feed-forward network with residule '''
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        # 12 Bert layers
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()

        # iterate over the 12 Bert layers
        for i, layer_module in enumerate(self.layer):
            # if self.output_hidden_states: # default False
            #     all_hidden_states = all_hidden_states + (hidden_states,)
            history_state = None if encoder_history_states is None else encoder_history_states[i] # default None

            layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            hidden_states = layer_outputs[0] # the output features [24, 95, 768]

            # if self.output_attentions: # default False
            #     all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        # if self.output_hidden_states: # default False
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # if self.output_hidden_states: # default False
        #     outputs = outputs + (all_hidden_states,)
        # if self.output_attentions: # default False
        #     outputs = outputs + (all_attentions,)

        return outputs  # outputs, (hidden states), (attentions)


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config) # one layer of transformer
        self.intermediate = BertIntermediate(config)  # [768 * 3072]
        self.output = BertOutput(config)              # [3072 * 768]

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)

        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


class BertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super(BertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        # self.img_feature_type = config.img_feature_type
        # if hasattr(config, 'use_img_layernorm'):
        #     self.use_img_layernorm = config.use_img_layernorm
        # else:
        #     self.use_img_layernorm = None

        # if config.img_feature_type == 'dis_code':
        #     self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
        #     self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        # elif config.img_feature_type == 'dis_code_t': # transpose
        #     self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
        #     self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        # elif config.img_feature_type == 'dis_code_scale': # scaled
        #     self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
        #     self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
        #     self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        # else:
        self.img_projection = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # if self.use_img_layernorm:
        #     self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    # def _resize_token_embeddings(self, new_num_tokens):
    #     old_embeddings = self.embeddings.word_embeddings
    #     new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    #     self.embeddings.word_embeddings = new_embeddings
    #     return self.embeddings.word_embeddings

    # def _prune_heads(self, heads_to_prune):
    #     """ Prunes heads of the model.
    #         heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
    #         See base class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # if head_mask is not None:
        #     if head_mask.dim() == 1:
        #         head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        #         head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1) # 12 heads
        #     elif head_mask.dim() == 2:
        #         head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        #     # switch to float if needed + fp16 compatibility
        #     head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        # else:
        #     head_mask = [None] * self.config.num_hidden_layers # 12 heads
        head_mask = [None] * self.config.num_hidden_layers # 12 heads

        # language embeddings [24, 55, 768]
        # embedding_output = self.embeddings(input_ids, position_ids=position_ids,
        #         token_type_ids=token_type_ids)
        language_features = input_ids
        # if encoder_history_states:
        #     assert img_feats is None, "Cannot take image features while using encoder history states"

        # if img_feats is not None:
        #     if self.img_feature_type == 'dis_code':
        #         code_emb = self.code_embeddings(img_feats)
        #         img_embedding_output = self.img_embedding(code_emb)
        #     elif self.img_feature_type == 'dis_code_t': # transpose
        #         code_emb = self.code_embeddings(img_feats)
        #         code_emb = code_emb.permute(0, 2, 1)
        #         img_embedding_output = self.img_embedding(code_emb)
        #     elif self.img_feature_type == 'dis_code_scale': # left scaled
        #         code_emb = self.code_embeddings(img_feats)
        #         img_embedding_output = self.img_embedding(code_emb)
        #     else: # faster r-cnn
        img_embedding_output = self.img_projection(img_feats) # [2054 * 768] projection
        # if self.use_img_layernorm:
        #     img_embedding_output = self.LayerNorm(img_embedding_output)
        # add dropout on image embedding
        img_embedding_output = self.dropout(img_embedding_output)

        # concatenate two embeddings
        concat_embedding_output = torch.cat((language_features, img_embedding_output), 1) # [24, 55+40, 768]

        ''' pass to the Transformer layers '''
        encoder_outputs = self.encoder(concat_embedding_output,
                extended_attention_mask, head_mask=head_mask,
                encoder_history_states=encoder_history_states) # [24, 95, 768]

        sequence_output = encoder_outputs[0]         # [24, 95, 768]
        pooled_output = self.pooler(sequence_output) # We "pool" the model by simply taking the hidden state corresponding to the first token [24, 768]

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs

class BertLanguageOnlyModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super(BertLanguageOnlyModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None):

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers # 12 heads

         # language embeddings [24, 55, 768]
        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)

        concat_embedding_output = embedding_output

        ''' pass to the Transformer layers '''
        encoder_outputs = self.encoder(concat_embedding_output,
                extended_attention_mask, head_mask=head_mask) # [24, 95, 768]

        sequence_output = encoder_outputs[0]         # [24, 95, 768]
        pooled_output = self.pooler(sequence_output) # We "pool" the model by simply taking the hidden state corresponding to the first token [24, 768]

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs


class LanguageBert(BertPreTrainedModel):
    """
    Modified from BertForMultipleChoice to support oscar training.
    """
    def __init__(self, config):
        super(LanguageBert, self).__init__(config)
        # self.loss_type = config.loss_type
        # if config.img_feature_dim > 0: # default for nlvr
        self.config = config
        if config.model_type == 'language':
            self.bert = BertLanguageOnlyModel(config) # LanuageOnlyBERT
        elif config.model_type == 'visual':
            self.bert = BertImgModel(config) # LanguageVisualBERT
        else:
            ModelTypeNotImplemented
        # else:
        #     self.bert = BertModel(config)  # original BERT

        # the classifier for downstream tasks
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # if hasattr(config, 'classifier'):
        #     if not hasattr(config, 'cls_hidden_scale'): config.cls_hidden_scale = 2
        #     if config.classifier == 'linear':
        #         self.classifier = nn.Linear(config.num_choice*config.hidden_size, self.config.num_labels)
        #     elif config.classifier == 'mlp':
        #         self.classifier = nn.Sequential(
        #             nn.Linear(config.num_choice*config.hidden_size, config.hidden_size*config.cls_hidden_scale),
        #             nn.ReLU(),
        #             nn.Linear(config.hidden_size*config.cls_hidden_scale, self.config.num_labels)
        #         )
        # else:
        #     self.classifier = nn.Linear(config.num_choice*config.hidden_size, self.config.num_labels)  # original

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, img_feats=None):
        # num_choices = input_ids.shape[1]
        #
        # flat_input_ids = input_ids.view(-1, input_ids.size(-1))                                                        # [24, 55]
        # flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None # [24, 55]
        # flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None # [24, 95]
        #
        # flat_img_feats = img_feats.view(-1, img_feats.size(-2), img_feats.size(-1)) if img_feats is not None else None # [24, 40, 2054]

        # if isinstance(self.bert, BertImgModel):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)
        # else:
        #     outputs = self.bert(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
        #                         attention_mask=flat_attention_mask, head_mask=head_mask)
        # outputs - the squence output

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # We "pool" the model by simply taking the hidden state corresponding to the first token [batch_size, 768]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if self.config.model_type == 'language':
            return sequence_output
        elif self.config.model_type == 'visual':
            return pooled_output
