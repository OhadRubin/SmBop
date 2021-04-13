from itertools import *
import torch
from allennlp.nn import util
from allennlp.nn.util import min_value_of_dtype, replace_masked_values
from functools import lru_cache


@lru_cache(maxsize=128)
def compute_op_idx(batch_size, seq_len, binary_op_count, unary_op_count, device):
    binary_op_ids = torch.arange(
        binary_op_count, dtype=torch.int64, device=device
    ).expand([batch_size, seq_len ** 2, binary_op_count])
    unary_op_ids = (
        torch.arange(unary_op_count, dtype=torch.int64, device=device) + binary_op_count
    ).expand([batch_size, seq_len, unary_op_count])

    frontier_op_ids = torch.cat(
        [
            binary_op_ids.reshape([batch_size, -1]),
            unary_op_ids.reshape([batch_size, -1]),
        ],
        dim=-1,
    )
    return frontier_op_ids


@lru_cache(maxsize=128)
def compute_beam_idx(batch_size, seq_len, binary_op_count, unary_op_count, device):
    binary_beam_idx = (
        torch.arange(seq_len ** 2, device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand([batch_size, seq_len ** 2, binary_op_count])
        .reshape([batch_size, -1])
    )
    l_binary_beam_idx = binary_beam_idx // seq_len
    r_binary_beam_idx = binary_beam_idx % seq_len
    unary_beam_idx = (
        torch.arange(seq_len, device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand([batch_size, seq_len, unary_op_count])
        .reshape([batch_size, -1])
    )
    l_beam_idx = torch.cat([l_binary_beam_idx, unary_beam_idx], dim=-1)
    r_beam_idx = torch.cat([r_binary_beam_idx, unary_beam_idx], dim=-1)
    return l_beam_idx, r_beam_idx


def batched_span_select(target: torch.Tensor, spans: torch.LongTensor) -> torch.Tensor:
    """
    The given `spans` of size `(batch_size, num_spans, 2)` indexes into the sequence
    dimension (dimension 2) of the target, which has size `(batch_size, sequence_length,
    embedding_size)`.
    This function returns segmented spans in the target with respect to the provided span indices.
    It does not guarantee element order within each span.
    # Parameters
    target : `torch.Tensor`, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : `torch.LongTensor`
        A 3 dimensional tensor of shape (batch_size, num_spans, 2) representing start and end
        indices (both inclusive) into the `sequence_length` dimension of the `target` tensor.
    # Returns
    span_embeddings : `torch.Tensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width, embedding_size]
        representing the embedded spans extracted from the batch flattened target tensor.
    span_mask: `torch.BoolTensor`
        A tensor with shape (batch_size, num_spans, max_batch_span_width) representing the mask on
        the returned span embeddings.
    """
    # both of shape (batch_size, num_spans, 1)
    span_starts, span_ends = spans.split(1, dim=-1)

    # shape (batch_size, num_spans, 1)
    # These span widths are off by 1, because the span ends are `inclusive`.
    span_widths = span_ends - span_starts

    # We need to know the maximum span width so we can
    # generate indices to extract the spans from the sequence tensor.
    # These indices will then get masked below, such that if the length
    # of a given span is smaller than the max, the rest of the values
    # are masked.
    max_batch_span_width = span_widths.max().item() + 1

    # Shape: (1, 1, max_batch_span_width)
    max_span_range_indices = util.get_range_vector(
        max_batch_span_width, util.get_device_of(target)
    ).view(1, 1, -1)
    #     print(max_batch_span_width)
    #     print(max_span_range_indices)
    # Shape: (batch_size, num_spans, max_batch_span_width)
    # This is a broadcasted comparison - for each span we are considering,
    # we are creating a range vector of size max_span_width, but masking values
    # which are greater than the actual length of the span.
    #
    # We're using <= here (and for the mask below) because the span ends are
    # inclusive, so we want to include indices which are equal to span_widths rather
    # than using it as a non-inclusive upper bound.
    span_mask = max_span_range_indices <= span_widths
    #     raw_span_indices = span_ends - max_span_range_indices
    raw_span_indices = span_starts + max_span_range_indices
    #     print(raw_span_indices)
    #     print(target.size())
    # We also don't want to include span indices which are less than zero,
    # which happens because some spans near the beginning of the sequence
    # have an end index < max_batch_span_width, so we add this to the mask here.
    span_mask = span_mask & (raw_span_indices < target.size(1))
    #     print(span_mask)
    #     span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()
    span_indices = raw_span_indices * span_mask
    #     print(span_indices)

    # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
    span_embeddings = util.batched_index_select(target, span_indices)

    return span_embeddings, span_mask


def shuffle(t):
    idx = torch.randperm(t.nelement())
    return t.view(-1)[idx].view(t.size())


def isin(key, query):
    key, _ = key.sort()
    a = torch.searchsorted(key, query, right=True)
    b = torch.searchsorted(key, query, right=False)
    return (a != b).float()


def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    """
    Replace the masked values in a tensor something really negative so that they won't
    affect a max operation.
    """
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))


def get_span_scores(
    span_start_logits: torch.Tensor, span_end_logits: torch.Tensor
) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.
    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(
        torch.ones((passage_length, passage_length), device=device)
    ).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    #     best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    #     span_start_indices = best_spans // passage_length
    #     span_end_indices = best_spans % passage_length
    #     return torch.stack([span_start_indices, span_end_indices], dim=-1)
    return valid_span_log_probs
