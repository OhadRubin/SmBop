import itertools
import json
import logging
import os
import time
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Dict

import allennlp
import torch
from allennlp.common.util import *
from allennlp.data import TokenIndexer, Vocabulary
from allennlp.models import Model
from allennlp.modules import (
    Seq2SeqEncoder,
    TextFieldEmbedder,
)

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.nn.util import masked_mean
from allennlp.training.metrics import Average
from anytree import PostOrderIter
from overrides import overrides

import smbop.utils.node_util as node_util
from smbop.eval_final.evaluation import evaluate_single
from smbop.utils import ra_postproc
from smbop.utils import vec_utils
from smbop.utils import hashing

logger = logging.getLogger(__name__)


@Model.register("smbop_parser")
class SmbopParser(Model):
    def __init__(
        self,
        experiment_name: str,
        vocab: Vocabulary,
        question_embedder: TextFieldEmbedder,
        schema_encoder: Seq2SeqEncoder,
        beam_encoder: Seq2SeqEncoder,
        tree_rep_transformer: Seq2SeqEncoder,
        utterance_augmenter: Seq2SeqEncoder,
        beam_summarizer: Seq2SeqEncoder,
        decoder_timesteps=9,
        beam_size=30,
        misc_params=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vocab)
        self._experiment_name = experiment_name
        self._misc_params = misc_params
        self.set_flags()
        self._utterance_augmenter = utterance_augmenter
        self._action_dim = beam_encoder.get_output_dim()
        self._beam_size = beam_size
        self._n_schema_leafs = 15
        self._num_values = 10

        self.tokenizer = TokenIndexer.by_name("pretrained_transformer")(
            model_name="Salesforce/grappa_large_jnt"
        )._allennlp_tokenizer.tokenizer

        if not self.cntx_reranker:
            self._noreranker_cntx_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2 * self._action_dim
            )
        if not self.utt_aug:
            self._nobeam_cntx_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2 * self._action_dim
            )
        self.activation_func = torch.nn.ReLU
        if self.lin_after_cntx:
            self.cntx_linear = torch.nn.Sequential(
                torch.nn.Linear(2 * self._action_dim, 4 * self._action_dim),
                torch.nn.Dropout(p=dropout),
                torch.nn.LayerNorm(4 * self._action_dim),
                self.activation_func(),
                torch.nn.Linear(4 * self._action_dim, 2 * self._action_dim),
            )
        if self.cntx_rep:
            self._cntx_rep_linear = torch.nn.Linear(
                in_features=self._action_dim, out_features=2 * self._action_dim
            )
        self._create_action_dicts()
        self.op_count = self.binary_op_count + self.unary_op_count
        self.xent = torch.nn.CrossEntropyLoss()

        self.type_embedding = torch.nn.Embedding(self.op_count, self._action_dim)
        self.summrize_vec = torch.nn.Embedding(
            num_embeddings=1, embedding_dim=self._action_dim
        )

        self.d_frontier = 2 * self._action_dim
        self.left_emb = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.d_frontier
        )
        self.right_emb = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.d_frontier
        )
        self.after_add = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
            torch.nn.Linear(self.d_frontier, self.d_frontier),
        )
        self._unary_frontier_embedder = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
            torch.nn.Linear(self.d_frontier, self.d_frontier),
        )

        self.op_linear = torch.nn.Linear(
            in_features=self.d_frontier, out_features=self.op_count
        )
        self.pre_op_linear = torch.nn.Sequential(
            torch.nn.Linear(self.d_frontier, self.d_frontier),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self.d_frontier),
            self.activation_func(),
        )

        assert (self._action_dim % 2) == 0
        self.vocab = vocab
        self._question_embedder = question_embedder
        self._schema_encoder = schema_encoder
        self._beam_encoder = beam_encoder
        self._beam_summarizer = beam_summarizer

        self._tree_rep_transformer = tree_rep_transformer

        self._decoder_timesteps = decoder_timesteps
        self._beam_size = beam_size
        self.q_emb_dim = question_embedder.get_output_dim()

        self.dropout_prob = dropout
        self._action_dim = beam_encoder.get_output_dim()
        self._span_score_func = torch.nn.Linear(self._action_dim, 2)
        self._pooler = BagOfEmbeddingsEncoder(embedding_dim=self._action_dim)

        self._rank_schema = torch.nn.Sequential(
            torch.nn.Linear(self._action_dim, self._action_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(self._action_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self._action_dim, 1),
        )
        self._rank_beam = torch.nn.Sequential(
            torch.nn.Linear(2 * self._action_dim, 2 * self._action_dim),
            torch.nn.Dropout(p=dropout),
            torch.nn.LayerNorm(2 * self._action_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(2 * self._action_dim, 1),
        )
        self._emb_to_action_dim = torch.nn.Linear(
            in_features=self.q_emb_dim,
            out_features=self._action_dim,
        )

        self._create_type_tensor()

        self._bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self._softmax = torch.nn.Softmax(dim=1)
        self._final_beam_acc = Average()
        self._reranker_acc = Average()
        self._spider_acc = Average()

        self._leafs_acc = Average()
        self._batch_size = -1
        self._device = None
        self._evaluate_func = partial(
            evaluate_single,
            db_dir=os.path.join("dataset", "database"),
            table_file=os.path.join("dataset", "tables.json"),
        )

    def set_flags(self):
        print(self._misc_params)
        self.is_oracle = self._misc_params.get("is_oracle", False)
        self.ranking_ratio = self._misc_params.get("ranking_ratio", 0.7)
        self.unique_reranker = self._misc_params.get("unique_reranker", False)
        self.cntx_reranker = self._misc_params.get("cntx_reranker", True)
        self.lin_after_cntx = self._misc_params.get("lin_after_cntx", False)
        self.utt_aug = self._misc_params.get("utt_aug", True)
        self.cntx_rep = self._misc_params.get("cntx_rep", False)
        self.add_residual_beam = self._misc_params.get("add_residual_beam", False)
        self.add_residual_reranker = self._misc_params.get(
            "add_residual_reranker", False
        )
        self.only_last_rerank = self._misc_params.get("only_last_rerank", False)
        self.oldlstm = self._misc_params.get("oldlstm", False)
        self.use_treelstm = self._misc_params.get("use_treelstm", False)
        self.disentangle_cntx = self._misc_params.get("disentangle_cntx", True)
        self.cntx_beam = self._misc_params.get("cntx_beam", True)
        self.uniquify = self._misc_params.get("uniquify", True)
        self.temperature = self._misc_params.get("temperature", 1.0)
        self.use_bce = self._misc_params["use_bce"]
        self.value_pred = self._misc_params.get("value_pred", True)
        self.debug = self._misc_params.get("debug", False)

        self.reuse_cntx_reranker = self._misc_params.get("reuse_cntx_reranker", True)
        self.should_rerank = self._misc_params.get("should_rerank", True)

    def _create_type_tensor(self):
        rule_tensor = [
            [[0] * len(self._type_dict) for _ in range(len(self._type_dict))]
            for _ in range(len(self._type_dict))
        ]
        if self.value_pred:
            RULES = node_util.RULES_values
        else:
            RULES = node_util.RULES_novalues

        rules = json.loads(RULES)
        for rule in rules:
            i, j_k = rule
            if len(j_k) == 0:
                continue
            elif len(j_k) == 2:
                j, k = j_k
            else:
                j, k = j_k[0], j_k[0]
            try:
                i, j, k = self._type_dict[i], self._type_dict[j], self._type_dict[k]
            except:
                continue
            rule_tensor[i][j][k] = 1
        self._rule_tensor = torch.tensor(rule_tensor)
        self._rule_tensor[self._type_dict["keep"]] = 1
        self._rule_tensor_flat = self._rule_tensor.flatten()
        self._op_count = self._rule_tensor.size(0)

        self._term_ids = [
            self._type_dict[i]
            for i in [
                "Project",
                "Orderby_desc",
                "Limit",
                "Groupby",
                "intersect",
                "except",
                "union",
                "Orderby_asc",
            ]
        ]
        self._term_tensor = torch.tensor(
            [1 if i in self._term_ids else 0 for i in range(len(self._type_dict))]
        )

    def _create_action_dicts(self):
        unary_ops = [
            "keep",
            "min",
            "count",
            "max",
            "avg",
            "sum",
            "Subquery",
            "distinct",
            "literal",
        ]

        binary_ops = [
            "eq",
            "like",
            "nlike",
            "add",
            "sub",
            "nin",
            "lte",
            "lt",
            "neq",
            "in",
            "gte",
            "gt",
            "And",
            "Or",
            "except",
            "union",
            "intersect",
            "Product",
            "Val_list",
            "Orderby_desc",
            "Orderby_asc",
            "Project",
            "Selection",
            "Limit",
            "Groupby",
        ]
        self.binary_op_count = len(binary_ops)
        self.unary_op_count = len(unary_ops)
        self._op_names = [
            k for k in itertools.chain(binary_ops, unary_ops, ["nan", "Table", "Value"])
        ]
        self._type_dict = OrderedDict({k: i for i, k in enumerate(self._op_names)})
        self.keep_id = self._type_dict["keep"]
        self._ACTIONS = {k: 1 for k in unary_ops}
        self._ACTIONS.update({k: 2 for k in binary_ops})
        self._ACTIONS = OrderedDict(self._ACTIONS)
        self._frontier_size = sum(self._beam_size ** n for n in self._ACTIONS.values())
        self.hasher = None
        self.flag_move_to_gpu = True

    def move_to_gpu(self, device):
        if self.flag_move_to_gpu:
            self._term_tensor = self._term_tensor.to(device)
            self._rule_tensor_flat = self._rule_tensor_flat.to(device)
            self._rule_tensor = self._rule_tensor.to(device)
            self.flag_move_to_gpu = False
    """
    

#enc
the question concatenated with the schema

#db_id
the id of the database schema we want to execute the query against

#schema_hash 
the hash of every schema string (applying dethash to every schema element) 

#schema_types
the type of every schema element (Value or Table), Value is either a Column or a literal.

#tree_obj
the AnyTree Node gold tree object after adding the hash attributes.

#gold_sql
the gold sql string.

#leaf_indices
makes it easier to pick the gold leaves during the oracle setup.

#entities
deprecated.

#orig_entities
used to reconstruct the tree for evaluation (this is added_values concatenated with the schema).

#is_gold_leaf
a boolean vector to tell if a given leaf is a gold leaf (i.e it corrosponds to a schema_hash that is in hash_gold_levelorder[0]).

#lengths
the length of the schema and the question, this is used to seperate them. 

#offsets
an array of size [batch_size, max_entity_token_length, 2] that contains the start and end indices for each schema token (and question, but that is inefficiet)
example:
given enc of [how,old,is,flights, flights, . ,start, flights, . ,end]
the output of batched_span_select given offsets would be:
[[how,pad,pad]
[[old,pad,pad]
..
[[flights,pad,pad]
[flights,.,start]
[flights,.,end]]

#relation
black box from ratsql

#depth
used to batch similar depth instances together. (see sorting keys in defaults.jsonnet)

#hash_gold_levelorder
An array of the gold hashes corrosponding to nodes in the gold tree
For example:
And  170816594
├── keep  -218759080
│   └── keep  -218759080
│       └── keep  -218759080
│           └── Value fictional_universe.type_of_fictional_setting -218759080
└── Join  -270677574
    ├── keep  55125446
    │   └── R  55125446
    │       └── Value fictional_universe.fictional_setting.setting_type 176689722
    └── Join  -149501965
        ├── keep  -94519500
        │   └── Value fictional_universe.fictional_setting.works_set_here -94519500
        └── literal  -26546860
            └── Value the atom cave raiders! 172249327
[[-218759080  176689722  -94519500  172249327]
 [-218759080   55125446  -94519500  -26546860]
 [-218759080   55125446 -149501965         -1]
 [-218759080 -270677574         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]
 [ 170816594         -1         -1         -1]]

#hash_gold_tree
root node hash

#span_hash
We apply dethash to every continuous span within the question, this results in a square matrix of size [batch_size, max_question_length, max_question_length].

#is_gold_span
a boolean vector to tell if a given span is a gold span (i.e it corrosponds to a span_hash that is in hash_gold_levelorder[0]).
    """
    def forward(
        self,
        enc,
        db_id,
        leaf_hash, #schema_hash
        leaf_types, #schema_types
        tree_obj=None,
        gold_sql=None,
        leaf_indices=None,
        entities=None,
        orig_entities=None,
        is_gold_leaf=None,
        lengths=None,
        offsets=None,
        relation=None,
        depth=None,
        hash_gold_levelorder=None,
        hash_gold_tree=None,
        span_hash=None,
        is_gold_span=None,
    ):

        total_start = time.time()
        outputs = {}
        beam_list = []
        item_list = []
        self._device = enc["tokens"]["token_ids"].device
        self.move_to_gpu(self._device)
        batch_size = len(db_id)
        self.hasher = hashing.Hasher(self._device)
        (
            embedded_schema,
            schema_mask,
            embedded_utterance,
            utterance_mask,
        ) = self._encode_utt_schema(enc, offsets, relation, lengths)
        batch_size, utterance_length, _ = embedded_utterance.shape
        start = time.time()
        loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        pre_loss = torch.tensor([0], dtype=torch.float32, device=self._device)
        vector_loss = torch.tensor(
            [0] * batch_size, dtype=torch.float32, device=self._device
        )
        # tree_sizes_vector = torch.tensor(
        #     [0] * batch_size, dtype=torch.float32, device=self._device
        # )

        tree_sizes_vector = torch.tensor(
            [1] * batch_size, dtype=torch.float32, device=self._device
        )
        if hash_gold_levelorder is not None:
            new_hash_gold_levelorder = hash_gold_levelorder.sort()[0].transpose(0, 1)
        if self.value_pred:
            span_scores, start_logits, end_logits = self.score_spans(
                embedded_utterance, utterance_mask
            )
            span_mask = torch.isfinite(span_scores).bool()
            final_span_scores = span_scores.clone()
            delta = final_span_scores.shape[-1] - span_hash.shape[-1]
            span_hash = torch.nn.functional.pad(
                span_hash,
                pad=(0, delta, 0, delta),
                mode="constant",
                value=-1,
            )
            if self.training:
                is_gold_span = torch.nn.functional.pad(
                    is_gold_span,
                    pad=(0, delta, 0, delta),
                    mode="constant",
                    value=0,
                )
                batch_idx, start_idx, end_idx = is_gold_span.nonzero().t()
                final_span_scores[
                    batch_idx, start_idx, end_idx
                ] = allennlp.nn.util.max_value_of_dtype(final_span_scores.dtype)

                is_span_end = is_gold_span.sum(-2).float()
                is_span_start = is_gold_span.sum(-1).float()

                span_start_probs = allennlp.nn.util.masked_log_softmax(
                    start_logits, utterance_mask.bool(), dim=1
                )
                span_end_probs = allennlp.nn.util.masked_log_softmax(
                    end_logits, utterance_mask.bool(), dim=1
                )

                vector_loss += (-span_start_probs * is_span_start.squeeze()).sum(-1) - (
                    span_end_probs * is_span_end.squeeze()
                ).sum(-1)
                tree_sizes_vector += 2 * is_span_start.squeeze().sum(-1)

            else:
                final_span_scores = span_scores
            

            _, leaf_span_mask, best_spans = allennlp.nn.util.masked_topk(
                final_span_scores.view([batch_size, -1]),
                span_mask.view([batch_size, -1]),
                self._num_values,
            )
            span_start_indices = best_spans // utterance_length
            span_end_indices = best_spans % utterance_length

            start_span_rep = allennlp.nn.util.batched_index_select(
                embedded_utterance.contiguous(), span_start_indices
            )
            end_span_rep = allennlp.nn.util.batched_index_select(
                embedded_utterance.contiguous(), span_end_indices
            )
            span_rep = (end_span_rep + start_span_rep) / 2
            leaf_span_rep = span_rep
            leaf_span_hash = allennlp.nn.util.batched_index_select(
                span_hash.reshape([batch_size, -1, 1]), best_spans
            ).reshape([batch_size, -1])
            leaf_span_types = torch.where(
                leaf_span_mask, self._type_dict["Value"], self._type_dict["nan"]
            ).int()

        leaf_schema_scores = self._rank_schema(embedded_schema)
        leaf_schema_scores = leaf_schema_scores / self.temperature
        if is_gold_leaf is not None:
            is_gold_leaf = torch.nn.functional.pad(
                is_gold_leaf,
                pad=(0, leaf_schema_scores.size(-2) - is_gold_leaf.size(-1)),
                mode="constant",
                value=0,
            )

        if self.training:
            final_leaf_schema_scores = leaf_schema_scores.clone()
            if not self.is_oracle:
                avg_leaf_schema_scores = allennlp.nn.util.masked_log_softmax(
                    final_leaf_schema_scores,
                    schema_mask.unsqueeze(-1).bool(),
                    dim=1,
                )
                loss_tensor = (
                    -avg_leaf_schema_scores * is_gold_leaf.unsqueeze(-1).float()
                )
                vector_loss += loss_tensor.squeeze().sum(-1)
                tree_sizes_vector += is_gold_leaf.squeeze().sum(-1).float()

            final_leaf_schema_scores = final_leaf_schema_scores.masked_fill(
                is_gold_leaf.bool().unsqueeze(-1),
                allennlp.nn.util.max_value_of_dtype(final_leaf_schema_scores.dtype),
            )
        else:
            final_leaf_schema_scores = leaf_schema_scores

        final_leaf_schema_scores = final_leaf_schema_scores.masked_fill(
            ~schema_mask.bool().unsqueeze(-1),
            allennlp.nn.util.min_value_of_dtype(final_leaf_schema_scores.dtype),
        )

        min_k = torch.clamp(schema_mask.sum(-1), 0, self._n_schema_leafs)
        _, leaf_schema_mask, top_beam_indices = allennlp.nn.util.masked_topk(
            final_leaf_schema_scores.squeeze(-1), mask=schema_mask.bool(), k=min_k
        )

        if self.is_oracle:

            leaf_indices = torch.nn.functional.pad(
                leaf_indices,
                pad=(0, self._n_schema_leafs - leaf_indices.size(-1)),
                mode="constant",
                value=-1,
            )
            leaf_schema_mask = leaf_indices >= 0
            final_leaf_indices = torch.abs(leaf_indices)

        else:
            final_leaf_indices = top_beam_indices
            
        leaf_schema_rep = allennlp.nn.util.batched_index_select(
            embedded_schema.contiguous(), final_leaf_indices
        )

        leaf_schema_hash = allennlp.nn.util.batched_index_select(
            leaf_hash.unsqueeze(-1), final_leaf_indices
        ).reshape([batch_size, -1])
        leaf_schema_types = (
            allennlp.nn.util.batched_index_select(
                leaf_types.unsqueeze(-1), final_leaf_indices
            )
            .reshape([batch_size, -1])
            .long()
        )
        if self.value_pred:
            beam_rep = torch.cat([leaf_schema_rep, leaf_span_rep], dim=-2)
            beam_hash = torch.cat([leaf_schema_hash, leaf_span_hash], dim=-1)
            beam_types = torch.cat([leaf_schema_types, leaf_span_types], dim=-1)
            beam_mask = torch.cat([leaf_schema_mask, leaf_span_mask], dim=-1)
            item_list.append(
                ra_postproc.ZeroItem(
                    beam_types,
                    final_leaf_indices,
                    span_start_indices,
                    span_end_indices,
                    orig_entities,
                    enc,
                    self.tokenizer,
                )
            )
        else:
            beam_rep = leaf_schema_rep
            beam_hash = leaf_schema_hash
            beam_types = leaf_schema_types
            beam_mask = leaf_schema_mask
            item_list.append(
                ra_postproc.ZeroItem(
                    beam_types,
                    final_leaf_indices,
                    None,
                    None,
                    orig_entities,
                    enc,
                    self.tokenizer,
                )
            )

        outputs["leaf_beam_hash"] = beam_hash
        outputs["hash_gold_levelorder"] = (batch_size*[None])
        # enc_list = [
        #     self.tokenizer.decode(enc["tokens"]["token_ids"][b].tolist())
        #     for b in range(batch_size)
        # ]

        for decoding_step in range(self._decoder_timesteps):
            batch_size, seq_len, _ = beam_rep.shape
            if self.utt_aug:
                enriched_beam_rep = self._augment_with_utterance(
                    embedded_utterance,
                    utterance_mask,
                    beam_rep,
                    beam_mask,
                    ctx=self._beam_encoder,
                )
            else:
                enriched_beam_rep = beam_rep
            if self.cntx_rep:
                beam_rep = enriched_beam_rep.contiguous()

            frontier_scores, frontier_mask = self.score_frontier(
                enriched_beam_rep, beam_rep, beam_mask
            )
            frontier_scores = frontier_scores / self.temperature
            l_beam_idx, r_beam_idx = vec_utils.compute_beam_idx(
                batch_size,
                seq_len,
                self.binary_op_count,
                self.unary_op_count,
                device=self._device,
            )
            frontier_op_ids = vec_utils.compute_op_idx(
                batch_size,
                seq_len,
                self.binary_op_count,
                self.unary_op_count,
                device=self._device,
            )

            frontier_hash = self.hash_frontier(
                beam_hash, frontier_op_ids, l_beam_idx, r_beam_idx
            )
            valid_op_mask = self.typecheck_frontier(
                beam_types, frontier_op_ids, l_beam_idx, r_beam_idx
            )
            frontier_mask = frontier_mask * valid_op_mask

            unique_frontier_scores = frontier_scores

            if self.training:
                with torch.no_grad():
                    is_levelorder_list = vec_utils.isin(
                        new_hash_gold_levelorder[decoding_step + 1], frontier_hash
                    )

                avg_frontier_scores = allennlp.nn.util.masked_log_softmax(
                    frontier_scores, frontier_mask.bool(), dim=1
                )
                loss_tensor = -avg_frontier_scores * is_levelorder_list.float()
                vector_loss += loss_tensor.squeeze().sum(-1)
                tree_sizes_vector += is_levelorder_list.bool().squeeze().sum(-1)

                unique_frontier_scores = unique_frontier_scores.masked_fill(
                    is_levelorder_list.bool(),
                    allennlp.nn.util.max_value_of_dtype(unique_frontier_scores.dtype),
                )

            beam_scores, beam_mask, beam_idx = allennlp.nn.util.masked_topk(
                unique_frontier_scores, mask=frontier_mask.bool(), k=self._beam_size
            )
            old_beam_types = beam_types.clone()

            beam_types = torch.gather(frontier_op_ids, -1, beam_idx)

            keep_indices = (beam_types == self.keep_id).nonzero().t().split(1)
            l_child_idx = torch.gather(l_beam_idx, -1, beam_idx)
            r_child_idx = torch.gather(r_beam_idx, -1, beam_idx)
            child_types = allennlp.nn.util.batched_index_select(
                old_beam_types.unsqueeze(-1), r_child_idx
            ).squeeze(-1)

            beam_rep = self._create_beam_rep(
                beam_rep, l_child_idx, r_child_idx, beam_types, keep_indices
            )

            beam_hash = torch.gather(frontier_hash, -1, beam_idx)
            if decoding_step == 1 and self.debug:
                failed_list, node_list, failed_set = get_failed_set(
                    beam_hash,
                    decoding_step,
                    tree_obj,
                    batch_size,
                    hash_gold_levelorder,
                )
                if failed_set:
                    print("hi")

            item_list.append(
                ra_postproc.Item(beam_types, l_child_idx, r_child_idx, beam_mask)
            )
            beam_types = torch.where(
                beam_types == self.keep_id, child_types, beam_types
            )
            beam_list.append(
                [
                    beam_hash.clone(),
                    beam_mask.clone(),
                    beam_types.clone(),
                    beam_scores.clone(),
                ]
            )

        if not self.training:
            (
                beam_hash_list,
                beam_mask_list,
                beam_type_list,
                beam_scores_list,
            ) = zip(*beam_list)
            beam_mask_tensor = torch.cat(beam_mask_list, dim=1)
            beam_type_tensor = torch.cat(beam_type_list, dim=1)

            is_final_mask = (
                self._term_tensor[beam_type_tensor].bool().to(beam_mask_tensor.device)
            )
            beam_mask_tensor = beam_mask_tensor * is_final_mask
            beam_hash_tensor = torch.cat(beam_hash_list, dim=1)
            beam_scores_tensor = torch.cat(beam_scores_list, dim=1)
            beam_scores_tensor = beam_scores_tensor
            beam_scores_tensor = beam_scores_tensor.masked_fill(
                ~beam_mask_tensor.bool(),
                allennlp.nn.util.min_value_of_dtype(beam_scores_tensor.dtype),
            )

        if self.training:
            pre_loss = (vector_loss / tree_sizes_vector).mean()

            loss = pre_loss.squeeze()
            assert not bool(torch.isnan(loss))
            outputs["loss"] = loss
            self._compute_validation_outputs(
                outputs,
                hash_gold_tree,
                beam_hash,
            )
            return outputs
        else:
            end = time.time()
            if tree_obj is not None:
                outputs["hash_gold_levelorder"] = [hash_gold_levelorder]+([None]*(batch_size-1))
            self._compute_validation_outputs(
                outputs,
                hash_gold_tree,
                beam_hash,
                is_gold_leaf=is_gold_leaf,
                top_beam_indices=top_beam_indices,
                db_id=db_id,
                beam_hash_tensor=beam_hash_tensor,
                beam_scores_tensor=beam_scores_tensor,
                gold_sql=gold_sql,
                item_list=item_list,
                inf_time=end - start,
                total_time=end - total_start,
            )
            return outputs

    def score_spans(self, embedded_utterance, utterance_mask):
        logits = self._span_score_func(embedded_utterance)
        logits = logits / self.temperature
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_logits = vec_utils.replace_masked_values_with_big_negative_number(
            start_logits, utterance_mask
        )
        end_logits = vec_utils.replace_masked_values_with_big_negative_number(
            end_logits, utterance_mask
        )
        span_scores = vec_utils.get_span_scores(start_logits, end_logits)
        return span_scores, start_logits, end_logits

    def _create_beam_rep(
        self, beam_rep, l_child_idx, r_child_idx, beam_types, keep_indices
    ):
        l_child_rep = allennlp.nn.util.batched_index_select(beam_rep, l_child_idx)
        r_child_rep = allennlp.nn.util.batched_index_select(beam_rep, r_child_idx)
        beam_type_rep = self.type_embedding(beam_types)
        beam_rep = torch.stack([beam_type_rep, l_child_rep, r_child_rep], dim=-2)
        batch_size, beam_size, _, emb_size = beam_rep.shape
        beam_rep = beam_rep.reshape([-1, 3, self._action_dim])
        mask = torch.ones([beam_rep.size(0), 3], dtype=torch.bool, device=self._device)
        beam_rep = self._tree_rep_transformer(inputs=beam_rep, mask=mask)
        beam_rep = self._pooler(beam_rep).reshape([batch_size, beam_size, emb_size])

        beam_rep[keep_indices] = r_child_rep[keep_indices].type(beam_rep.dtype)
        return beam_rep

    def _compute_validation_outputs(
        self,
        outputs,
        hash_gold_tree,
        beam_hash,
        **kwargs,
    ):
        batch_size = beam_hash.size(0)
        final_beam_acc_list = []
        reranker_acc_list = []
        spider_acc_list = []
        leaf_acc_list = []
        sql_list = []
        tree_list = []
        beam_scores_el_list = []
        if hash_gold_tree is not None:
            for gs, fa in zip(hash_gold_tree, beam_hash.tolist()):
                acc = int(gs) in fa
                self._final_beam_acc(int(acc))
                final_beam_acc_list.append(bool(acc))

        if not self.training:

            if (
                kwargs["is_gold_leaf"] is not None
                and kwargs["top_beam_indices"] is not None
            ):
                for top_beam_indices_el, is_gold_leaf_el in zip(
                    kwargs["top_beam_indices"], kwargs["is_gold_leaf"]
                ):
                    is_gold_leaf_idx = is_gold_leaf_el.nonzero().squeeze().tolist()
                    leaf_acc = int(
                        all([x in top_beam_indices_el for x in is_gold_leaf_idx])
                    )
                    leaf_acc_list.append(leaf_acc)
                    self._leafs_acc(leaf_acc)

            # TODO: change this!! this causes bugs!
            for b in range(batch_size):
                beam_scores_el = kwargs["beam_scores_tensor"][b]
                beam_scores_el[
                    : -self._beam_size
                ] = allennlp.nn.util.min_value_of_dtype(beam_scores_el.dtype)
                beam_scores_el_list.append(beam_scores_el)
                top_idx = int(beam_scores_el.argmax())
                tree_copy = ""
                try:
                    items = kwargs["item_list"][: (top_idx // self._beam_size) + 2]

                    tree_res = ra_postproc.reconstruct_tree(
                        self._op_names,
                        self.binary_op_count,
                        b,
                        top_idx % self._beam_size,
                        items,
                        len(items) - 1,
                        self._n_schema_leafs,
                    )
                    tree_copy = deepcopy(tree_res)
                    sql = ra_postproc.ra_to_sql(tree_res)
                except:
                    print("Could not reconstruct SQL from RA tree")
                    sql = ""
                spider_acc = 0
                reranker_acc = 0

                outputs["inf_time"] = [kwargs["inf_time"]]+([None]*(batch_size-1))
                outputs["total_time"] = [kwargs["total_time"]] + \
                    ([None]*(batch_size-1))

                if hash_gold_tree is not None:
                    try:
                        reranker_acc = int(
                            kwargs["beam_hash_tensor"][b][top_idx]
                            == int(hash_gold_tree[b])
                        )

                        gold_sql = kwargs["gold_sql"][b]
                        db_id = kwargs["db_id"][b]
                        spider_acc = int(self._evaluate_func(gold_sql, sql, db_id))
                    except:
                        print("EM evaluation failed")

                reranker_acc_list.append(reranker_acc)
                self._reranker_acc(reranker_acc)
                self._spider_acc(spider_acc)
                sql_list.append(sql)
                tree_list.append(tree_copy)
                spider_acc_list.append(spider_acc)
            outputs["beam_scores"] = beam_scores_el_list
            outputs["beam_encoding"] = [kwargs["item_list"]]+([None]*(batch_size-1))
            outputs["beam_hash"] = [kwargs["beam_hash_tensor"]]+([None]*(batch_size-1))
            # outputs["gold_hash"] = hash_gold_tree or ([None]*batch_size)
            if hash_gold_tree is not None:
                outputs["gold_hash"] = hash_gold_tree
            else:
                outputs["gold_hash"] = [hash_gold_tree] + ([None]*(batch_size-1))
            outputs["reranker_acc"] = reranker_acc_list
            outputs["spider_acc"] = spider_acc_list
            outputs["sql_list"] = sql_list
            outputs["tree_list"] = tree_list
        outputs["final_beam_acc"] = final_beam_acc_list or ([None]*batch_size)
        outputs["leaf_acc"] = leaf_acc_list or ([None]*batch_size)

    def _augment_with_utterance(
        self,
        embedded_utterance,
        utterance_mask,
        beam_rep,
        beam_mask,
        ctx=None,
    ):
        assert ctx

        if self.disentangle_cntx:
            enriched_beam_rep = self._utterance_augmenter(
                beam_rep, embedded_utterance, ctx_att_mask=utterance_mask
            )[0]
            if self.cntx_beam:
                enriched_beam_rep = ctx(inputs=enriched_beam_rep, mask=beam_mask.bool())
        else:
            encoder_input = torch.cat([embedded_utterance, beam_rep], dim=1)
            input_mask = torch.cat([utterance_mask.bool(), beam_mask.bool()], dim=-1)
            encoder_output = ctx(inputs=encoder_input, mask=input_mask)
            _, enriched_beam_rep = torch.split(
                encoder_output, [utterance_mask.size(-1), beam_mask.size(-1)], dim=1
            )

        return enriched_beam_rep

    def emb_q(self, enc):
        pad_dim = enc["tokens"]["mask"].size(-1)
        if pad_dim > 512:
            for key in enc["tokens"].keys():
                enc["tokens"][key] = enc["tokens"][key][:, :512]

            embedded_utterance_schema = self._question_embedder(enc)
        else:
            embedded_utterance_schema = self._question_embedder(enc)

        return embedded_utterance_schema

    def _encode_utt_schema(self, enc, offsets, relation, lengths):
        embedded_utterance_schema = self.emb_q(enc)

        (
            embedded_utterance_schema,
            embedded_utterance_schema_mask,
        ) = vec_utils.batched_span_select(embedded_utterance_schema, offsets)
        embedded_utterance_schema = masked_mean(
            embedded_utterance_schema,
            embedded_utterance_schema_mask.unsqueeze(-1),
            dim=-2,
        )

        relation_mask = (relation >= 0).float()  # TODO: fixme
        torch.abs(relation, out=relation)
        embedded_utterance_schema = self._emb_to_action_dim(embedded_utterance_schema)
        enriched_utterance_schema = self._schema_encoder(
            embedded_utterance_schema, relation.long(), relation_mask
        )

        utterance_schema, utterance_schema_mask = vec_utils.batched_span_select(
            enriched_utterance_schema, lengths
        )
        utterance, schema = torch.split(utterance_schema, 1, dim=1)
        utterance_mask, schema_mask = torch.split(utterance_schema_mask, 1, dim=1)
        utterance_mask = torch.squeeze(utterance_mask, 1)
        schema_mask = torch.squeeze(schema_mask, 1)
        embedded_utterance = torch.squeeze(utterance, 1)
        schema = torch.squeeze(schema, 1)
        return schema, schema_mask, embedded_utterance, utterance_mask

    def score_frontier(self, enriched_beam_rep, beam_rep, beam_mask):
        if self.cntx_rep:
            beam_rep = self._cntx_rep_linear(enriched_beam_rep)
        else:
            if self.utt_aug:
                beam_rep = torch.cat([enriched_beam_rep, beam_rep], dim=-1)
                if self.lin_after_cntx:
                    beam_rep = self.cntx_linear(beam_rep)
            else:
                beam_rep = self._nobeam_cntx_linear(beam_rep)

        batch_size, seq_len, emb_size = beam_rep.shape

        left = self.left_emb(beam_rep.reshape([batch_size, seq_len, 1, emb_size]))
        right = self.right_emb(beam_rep.reshape([batch_size, 1, seq_len, emb_size]))
        binary_ops_reps = self.after_add(left + right)
        binary_ops_reps = binary_ops_reps.reshape(-1, seq_len ** 2, self.d_frontier)
        unary_ops_reps = self._unary_frontier_embedder(beam_rep)
        pre_frontier_rep = torch.cat([binary_ops_reps, unary_ops_reps], dim=1)
        pre_frontier_rep = self.pre_op_linear(pre_frontier_rep)

        base_frontier_scores = self.op_linear(pre_frontier_rep)
        binary_frontier_scores, unary_frontier_scores = torch.split(
            base_frontier_scores, [seq_len ** 2, seq_len], dim=1
        )
        binary_frontier_scores, _ = torch.split(
            binary_frontier_scores, [self.binary_op_count, self.unary_op_count], dim=2
        )
        _, unary_frontier_scores = torch.split(
            unary_frontier_scores, [self.binary_op_count, self.unary_op_count], dim=2
        )
        frontier_scores = torch.cat(
            [
                binary_frontier_scores.reshape([batch_size, -1]),
                unary_frontier_scores.reshape([batch_size, -1]),
            ],
            dim=-1,
        )
        binary_mask = torch.einsum("bi,bj->bij", beam_mask, beam_mask)
        binary_mask = binary_mask.view([beam_mask.shape[0], -1]).unsqueeze(-1)
        binary_mask = binary_mask.expand(
            [batch_size, seq_len ** 2, self.binary_op_count]
        ).reshape(batch_size, -1)
        unary_mask = (
            beam_mask.clone()
            .unsqueeze(-1)
            .expand([batch_size, seq_len, self.unary_op_count])
            .reshape(batch_size, -1)
        )
        frontier_mask = torch.cat([binary_mask, unary_mask], dim=-1)

        return frontier_scores, frontier_mask

    def hash_frontier(self, beam_hash, frontier_op_ids, l_beam_idx, r_beam_idx):
        r_hash = (
            allennlp.nn.util.batched_index_select(beam_hash.unsqueeze(-1), r_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        l_hash = (
            allennlp.nn.util.batched_index_select(beam_hash.unsqueeze(-1), l_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        tmp = l_hash.clone()
        frontier_hash = self.set_hash(
            frontier_op_ids.clone().reshape(-1), l_hash, r_hash
        ).long()
        frontier_hash = torch.where(
            frontier_op_ids.reshape(-1) == self.keep_id, tmp, frontier_hash
        )
        frontier_hash = frontier_hash.reshape(r_beam_idx.size())
        return frontier_hash

    def typecheck_frontier(self, beam_types, frontier_op_ids, l_beam_idx, r_beam_idx):
        batch_size, frontier_size = frontier_op_ids.shape

        r_types = (
            allennlp.nn.util.batched_index_select(beam_types.unsqueeze(-1), r_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        l_types = (
            allennlp.nn.util.batched_index_select(beam_types.unsqueeze(-1), l_beam_idx)
            .squeeze(-1)
            .reshape(-1)
        )
        indices_into = (
            self._op_count * self._op_count * frontier_op_ids.view(-1)
            + self._op_count * l_types
            + r_types
        )
        valid_ops = self._rule_tensor_flat[indices_into].reshape(
            [batch_size, frontier_size]
        )
        return valid_ops

    def set_hash(self, parent, a, b):
        a <<= 28
        b >>= 1
        a = a.add_(b)
        parent <<= 56
        a = a.add_(parent)
        a *= self.hasher.tensor2
        # TODO check lgu-lgm hashing instead of this:
        a = a.fmod_(self.hasher.tensor1)
        return a

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        out = {
            "final_beam_acc": self._final_beam_acc.get_metric(reset),
        }
        if not self.training:
            out["spider"] = self._spider_acc.get_metric(reset)
            out["reranker"] = self._reranker_acc.get_metric(reset)
            out["leafs_acc"] = self._leafs_acc.get_metric(reset)
            # out['self._spider_acc._count'] = self._spider_acc._count
        return out


def get_failed_set(
    beam_hash, decoding_step, tree_obj, batch_size, hash_gold_levelorder
):
    failed_set = []
    failed_list = []
    node_list = []
    for b in range(batch_size):
        node_list.append(node_util.print_tree(tree_obj[b]))
        node_dict = {node.hash: node for node in PostOrderIter(tree_obj[b])}
        batch_set = (
            set(hash_gold_levelorder[b][decoding_step + 1].tolist())
            - set(beam_hash[b].tolist())
        ) - {-1}
        failed_list.append([node_dict[set_el] for set_el in batch_set])
        failed_set.extend([node_dict[set_el] for set_el in batch_set])
    return failed_list, node_list, failed_set
