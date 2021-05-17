from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, TokenIndexer, Field, Instance
from allennlp.data.fields import TextField, ListField, IndexField, MetadataField
from allennlp.data.fields import (
    TextField,
    ListField,
    IndexField,
    MetadataField,
    ArrayField,
)

import anytree
from anytree.search import *
from collections import defaultdict
from overrides import overrides
from time import time
from typing import Dict
from smbop.utils import moz_sql_parser as msp

import smbop.utils.node_util as node_util
import smbop.utils.hashing as hashing
import smbop.utils.ra_preproc as ra_preproc
from anytree import Node, LevelOrderGroupIter
import dill
import itertools
from collections import defaultdict, OrderedDict
import json
import logging
import numpy as np
import os
from smbop.utils.replacer import Replacer
import time
from smbop.dataset_readers.enc_preproc import *
import smbop.dataset_readers.disamb_sql as disamb_sql
from smbop.utils.cache import TensorCache

logger = logging.getLogger(__name__)


@DatasetReader.register("smbop")
class SmbopSpiderDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = True,
        question_token_indexers: Dict[str, TokenIndexer] = None,
        keep_if_unparsable: bool = True,
        tables_file: str = None,
        dataset_path: str = "dataset/database",
        cache_directory: str = "cache/train",
        include_table_name_in_column=True,
        fix_issue_16_primary_keys=False,
        qq_max_dist=2,
        cc_max_dist=2,
        tt_max_dist=2,
        max_instances=10000000,
        decoder_timesteps=9,
        limit_instances=-1,
        value_pred=True,
        use_longdb=True,
    ):
        super().__init__(
            # lazy=lazy,
            # cache_directory=cache_directory,
            # max_instances=max_instances,
            #  manual_distributed_sharding=True,
            # manual_multi_process_sharding=True,
        )
        self.cache_directory = cache_directory
        self.cache = TensorCache(cache_directory)
        self.value_pred = value_pred
        self._decoder_timesteps = decoder_timesteps
        self._max_instances = max_instances
        self.limit_instances = limit_instances
        self.load_less = limit_instances!=-1

        self._utterance_token_indexers = question_token_indexers

        self._tokenizer = self._utterance_token_indexers["tokens"]._allennlp_tokenizer
        self.cls_token = self._tokenizer.tokenize("a")[0]
        self.eos_token = self._tokenizer.tokenize("a")[-1]
        self._keep_if_unparsable = keep_if_unparsable

        self._tables_file = tables_file
        self._dataset_path = dataset_path

        # ratsql
        self.enc_preproc = EncPreproc(
            tables_file,
            dataset_path,
            include_table_name_in_column,
            fix_issue_16_primary_keys,
            qq_max_dist,
            cc_max_dist,
            tt_max_dist,
            use_longdb,
        )
        self._create_action_dicts()
        self.replacer = Replacer(tables_file)

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
        self.hasher = hashing.Hasher("cpu")

    def _init_fields(self, tree_obj):
        tree_obj = node_util.add_max_depth_att(tree_obj)
        tree_obj = node_util.tree2maxdepth(tree_obj)
        tree_obj = self.hasher.add_hash_att(tree_obj, self._type_dict)
        hash_gold_tree = tree_obj.hash
        hash_gold_levelorder = []
        for tree_list in LevelOrderGroupIter(tree_obj):
            hash_gold_levelorder.append([tree.hash for tree in tree_list])

        pad_el = hash_gold_levelorder[0]
        for i in range(self._decoder_timesteps - len(hash_gold_levelorder) + 2):
            hash_gold_levelorder.insert(0, pad_el)
        hash_gold_levelorder = hash_gold_levelorder[::-1]
        max_size = max(len(level) for level in hash_gold_levelorder)
        for level in hash_gold_levelorder:
            level.extend([-1] * (max_size - len(level)))
        hash_gold_levelorder = np.array(hash_gold_levelorder)
        return (
            hash_gold_levelorder,
            hash_gold_tree,
        )

    def process_instance(self, instance: Instance, index: int):
        return instance

    @overrides
    def _read(self, file_path: str):
        if file_path.endswith(".json"):
            yield from self._read_examples_file(file_path)
        else:
            raise ConfigurationError(f"Don't know how to read filetype of {file_path}")

    def _read_examples_file(self, file_path: str):
        # cache_dir = os.path.join("cache", file_path.split("/")[-1])

        cnt = 0
        cache_buffer = []
        cont_flag = True
        sent_set = set()
        for total_cnt,ins in self.cache:
            if cnt >= self._max_instances:
                break
            if ins is not None:
                yield ins
                cnt += 1
            sent_set.add(total_cnt)
            if self.load_less and len(sent_set) > self.limit_instances:
                cont_flag = False
                break

        if cont_flag:
            with open(file_path, "r") as data_file:
                json_obj = json.load(data_file)
                for total_cnt, ex in enumerate(json_obj):
                    if cnt >= self._max_instances:
                        break
                    if len(cache_buffer)>50:
                        self.cache.write(cache_buffer)
                        cache_buffer = []
                    if total_cnt in sent_set:
                        continue
                    else:    
                        ins = self.create_instance(ex)
                        cache_buffer.append([total_cnt, ins])
                    if ins is not None:
                        yield ins
                        cnt +=1
            self.cache.write(cache_buffer)


    def process_instance(self, instance: Instance, index: int):
        return instance

    def create_instance(self,ex):
        sql = None
        sql_with_values = None

        if "query_toks" in ex:
            try:
                ex = disamb_sql.fix_number_value(ex)
                sql = disamb_sql.disambiguate_items(
                    ex["db_id"],
                    ex["query_toks_no_value"],
                    self._tables_file,
                    allow_aliases=False,
                )
                sql_with_values = disamb_sql.sanitize(ex["query"])
            except Exception as e:
                # there are two examples in the train set that are wrongly formatted, skip them
                print(f"error with {ex['query']}")
                return None

        ins = self.text_to_instance(
            utterance=ex["question"],
            db_id=ex["db_id"],
            sql=sql,
            sql_with_values=sql_with_values,
        )
        return ins


    def text_to_instance(
        self, utterance: str, db_id: str, sql=None, sql_with_values=None
    ):
        fields: Dict[str, Field] = {
            "db_id": MetadataField(db_id),
        }

        tokenized_utterance = self._tokenizer.tokenize(utterance)
        has_gold = sql is not None

        if has_gold:
            try:
                tree_dict = msp.parse(sql)
                tree_dict_values = msp.parse(sql_with_values)
            except msp.ParseException as e:
                print(f"could'nt create AST for:  {sql}")
                return None
            tree_obj = ra_preproc.ast_to_ra(tree_dict["query"])
            tree_obj_values = ra_preproc.ast_to_ra(tree_dict_values["query"])

            arit_list = anytree.search.findall(
                tree_obj, filter_=lambda x: x.name in ["sub", "add"]
            )  # TODO: fixme
            haslist_list = anytree.search.findall(
                tree_obj,
                filter_=lambda x: hasattr(x, "val") and isinstance(x.val, list),
            )
            if arit_list or haslist_list:
                print(f"could'nt create RA for:  {sql}")
                
                return None
            if self.value_pred:
                for a, b in zip(tree_obj_values.leaves, tree_obj.leaves):
                    if b.name == "Table" or ("." in str(b.val)):
                        continue
                    b.val = a.val
                    if (
                        isinstance(a.val, int) or isinstance(a.val, float)
                    ) and b.parent.name == "literal":
                        parent_node = b.parent
                        parent_node.children = []
                        parent_node.name = "Value"
                        parent_node.val = b.val

            for leaf in tree_obj.leaves:
                leaf.val = self.replacer.pre(leaf.val, db_id)
                if not self.value_pred and node_util.is_number(leaf.val):
                    leaf.val = "value"

            leafs = list(set(node_util.get_leafs(tree_obj)))
            hash_gold_levelorder, hash_gold_tree = self._init_fields(tree_obj)

            fields.update(
                {
                    "hash_gold_levelorder": ArrayField(
                        hash_gold_levelorder, padding_value=-1, dtype=np.int64
                    ),
                    "hash_gold_tree": ArrayField(
                        np.array(hash_gold_tree), padding_value=-1, dtype=np.int64
                    ),
                    "gold_sql": MetadataField(sql_with_values),
                    "tree_obj": MetadataField(tree_obj),
                }
            )

        desc = self.enc_preproc.get_desc(tokenized_utterance, db_id)
        entities, added_values, relation = self.extract_relation(desc)


        question_concated = [[x] for x in tokenized_utterance[1:-1]]
        schema_tokens_pre, schema_tokens_pre_mask = table_text_encoding(
            entities[len(added_values) + 1 :]
        )

        schema_size = len(entities)
        schema_tokens_pre = added_values + ["*"] + schema_tokens_pre

        schema_tokens = [
            [y for y in x if y.text not in ["_"]]
            for x in [self._tokenizer.tokenize(x)[1:-1] for x in schema_tokens_pre]
        ]

        entities_as_leafs = [x.split(":")[0] for x in entities[len(added_values) + 1 :]]
        entities_as_leafs = added_values + ["*"] + entities_as_leafs
        orig_entities = [self.replacer.post(x, db_id) for x in entities_as_leafs]
        entities_as_leafs_hash, entities_as_leafs_types = self.hash_schema(
            entities_as_leafs, added_values
        )

        fields.update(
            {
                "relation": ArrayField(relation, padding_value=-1, dtype=np.int32),
                "entities": MetadataField(entities_as_leafs),
                 "orig_entities": MetadataField(orig_entities),
                 "leaf_hash": ArrayField(
                    entities_as_leafs_hash, padding_value=-1, dtype=np.int64
                ),
                "leaf_types": ArrayField(
                    entities_as_leafs_types,
                    padding_value=self._type_dict["nan"],
                    dtype=np.int32,
                )
            })

        if has_gold:
            leaf_indices, is_gold_leaf, depth = self.is_gold_leafs(
                tree_obj, leafs, schema_size, entities_as_leafs
            )
            fields.update(
                {
                    "is_gold_leaf": ArrayField(
                        is_gold_leaf, padding_value=0, dtype=np.int32
                    ),
                    "leaf_indices": ArrayField(
                        leaf_indices, padding_value=-1, dtype=np.int32
                    ),
                    "depth": ArrayField(depth, padding_value=0, dtype=np.int32),
                }
            )

        utt_len = len(tokenized_utterance[1:-1])
        if self.value_pred:
            span_hash_array = self.hash_spans(tokenized_utterance)
            fields["span_hash"] = ArrayField(
                span_hash_array, padding_value=-1, dtype=np.int64
            )

        if has_gold and self.value_pred:
            value_list = np.array(
                [self.hash_text(x) for x in node_util.get_literals(tree_obj)],
                dtype=np.int64,
            )
            is_gold_span = np.isin(span_hash_array.reshape([-1]), value_list).reshape(
                [utt_len, utt_len]
            )
            fields["is_gold_span"] = ArrayField(
                is_gold_span, padding_value=False, dtype=np.bool
            )

        enc_field_list = []
        offsets = []
        mask_list = (
            [False]
            + ([True] * len(question_concated))
            + [False]
            + ([True] * len(added_values))
            + [True]
            + schema_tokens_pre_mask
            + [False]
        )
        for mask, x in zip(
            mask_list,
            [[self.cls_token]]
            + question_concated
            + [[self.eos_token]]
            + schema_tokens
            + [[self.eos_token]],
        ):
            start_offset = len(enc_field_list)
            enc_field_list.extend(x)
            if mask:
                offsets.append([start_offset, len(enc_field_list) - 1])

        fields["lengths"] = ArrayField(
            np.array(
                [
                    [0, len(question_concated) - 1],
                    [len(question_concated), len(question_concated) + schema_size - 1],
                ]
            ),
            dtype=np.int32,
        )
        fields["offsets"] = ArrayField(
            np.array(offsets), padding_value=0, dtype=np.int32
        )
        fields["enc"] = TextField(enc_field_list)

        ins = Instance(fields)
        return ins

    def extract_relation(self, desc):
        def parse_col(col_list):
            col_type = col_list[0]
            col_name, table = "_".join(col_list[1:]).split("_<table-sep>_")
            return f'{table}.{col_name}:{col_type.replace("<type: ","")[:-1]}'

        question_concated = [x for x in desc["question"]]
        col_concated = [parse_col(x) for x in desc["columns"]]
        table_concated = ["_".join(x).lower() for x in desc["tables"]]
        enc = question_concated + col_concated + table_concated
        relation = self.enc_preproc.compute_relations(
            desc,
            len(enc),
            len(question_concated),
            len(col_concated),
            range(len(col_concated) + 1),
            range(len(table_concated) + 1),
        )
        unsorted_entities = col_concated + table_concated
        rel_dict = defaultdict(dict)
        # can do this with one loop
        for i, x in enumerate(list(range(len(question_concated))) + unsorted_entities):
            for j, y in enumerate(
                list(range(len(question_concated))) + unsorted_entities
            ):
                rel_dict[x][y] = relation[i, j]
        entities_sorted = sorted(list(enumerate(unsorted_entities)), key=lambda x: x[1])
        entities = [x[1] for x in entities_sorted]
        if self.value_pred:
            added_values = [
                "1",
                "2",
                "3",
                "4",
                "5",
                "yes",
                "no",
                "y",
                "t",
                "f",
                "m",
                "n",
                "null",
            ]
        else:
            added_values = ["value"]
        entities = added_values + entities
        new_enc = list(range(len(question_concated))) + entities
        new_relation = np.zeros([len(new_enc), len(new_enc)])
        for i, x in enumerate(new_enc):
            for j, y in enumerate(new_enc):
                if y in added_values or x in added_values:
                    continue
                new_relation[i][j] = rel_dict[x][y]
        return entities, added_values, new_relation

    def is_gold_leafs(self, tree_obj, leafs, schema_size, entities_as_leafs):
        enitities_leaf_dict = {ent: i for i, ent in enumerate(entities_as_leafs)}
        indices = []
        for leaf in leafs:
            leaf = str(leaf).lower()
            if leaf in enitities_leaf_dict:
                indices.append(enitities_leaf_dict[leaf])
        is_gold_leaf = np.array(
            [1 if (i in indices) else 0 for i in range(schema_size)]
        )
        indices = np.array(indices)
        depth = np.array([1] * max([leaf.depth for leaf in tree_obj.leaves]))
        return indices, is_gold_leaf, depth

    def hash_schema(self, leaf_text, added_values=None):
        beam_hash = []
        beam_types = []

        for leaf in leaf_text:
            leaf = leaf.strip()
            # TODO: fix this
            if (len(leaf.split(".")) == 2) or ("*" == leaf) or leaf in added_values:
                leaf_node = Node("Value", val=leaf)
                type_ = self._type_dict["Value"]
            else:
                leaf_node = Node("Table", val=leaf)
                type_ = self._type_dict["Table"]
            leaf_node = self.hasher.add_hash_att(leaf_node, self._type_dict)
            beam_hash.append(leaf_node.hash)
            beam_types.append(type_)
        beam_hash = np.array(beam_hash, dtype=np.int64)
        beam_types = np.array(beam_types, dtype=np.int32)
        return beam_hash, beam_types

    def hash_text(self, text):
        return self.hasher.set_hash([self._type_dict["Value"], hashing.dethash(text)])

    def hash_spans(self, tokenized_utterance):
        utt_idx = [x.text_id for x in tokenized_utterance[1:-1]]
        utt_len = len(utt_idx)
        span_hash_array = -np.ones([utt_len, utt_len], dtype=int)
        for i_ in range(utt_len):
            for j_ in range(utt_len):
                if i_ <= j_:
                    span_text = self._tokenizer.tokenizer.decode(utt_idx[i_ : j_ + 1])
                    span_hash_array[i_, j_] = self.hash_text(span_text)
        return span_hash_array

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["enc"].token_indexers = self._utterance_token_indexers


def table_text_encoding(entity_text_list):
    token_list = []
    mask_list = []
    for i, curr in enumerate(entity_text_list):
        if ":" in curr:  # col
            token_list.append(curr)
            if (i + 1) < len(entity_text_list) and ":" in entity_text_list[i + 1]:
                token_list.append(",")
            else:
                token_list.append(")\n")
            mask_list.extend([True, False])
        else:
            token_list.append(curr)
            token_list.append("(")
            mask_list.extend([True, False])

    return token_list, mask_list
