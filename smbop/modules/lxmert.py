import torch
from torch import nn
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
import math
import allennlp.nn.util as ai2_util


@Seq2SeqEncoder.register("cross_attention")
class LxmertCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        ctx_dim,
        hidden_dropout_prob,
    ):
        super().__init__()
        self.att = LxmertAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim
        )
        self.output = LxmertAttentionOutput(hidden_size, hidden_dropout_prob)

    def forward(
        self, input_tensor, ctx_tensor, ctx_att_mask=None, output_attentions=False
    ):
        output = self.att(
            input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions
        )
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (
            (attention_output, attention_probs)
            if output_attentions
            else (attention_output,)
        )
        return outputs


class LxmertAttention(nn.Module):
    def __init__(
        self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        # if ctx_dim is None:
        self.ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, context, attention_mask=None, output_attentions=False
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            # print(attention_scores.shape)
            # print(attention_mask.shape)
            attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores.clone().masked_fill(
                ~attention_mask, ai2_util.min_value_of_dtype(attention_scores.dtype)
            )
            # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class LxmertAttentionOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
