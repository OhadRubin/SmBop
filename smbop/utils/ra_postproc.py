from functools import reduce
from itertools import *
from anytree import Node
import copy
from anytree.search import *
import re
import smbop.utils.node_util as node_util


else_dict = {
    "Selection": " WHERE ",
    "Groupby": " GROUP BY ",
    "Limit": " LIMIT ",
    "Having": " HAVING ",
}

pred_dict = {
    "eq": " = ",
    "like": " LIKE ",
    "nin": " NOT IN ",
    "lte": " <= ",
    "lt": "<",
    "neq": " != ",
    "in": " IN ",
    "gte": " >= ",
    "gt": " > ",
    "And": " AND ",
    "Or": " OR ",
    "except": " EXCEPT ",
    "union": " UNION ",
    "intersect": " INTERSECT ",
    "Val_list": " , ",
    "Product": " , ",
}


def wrap_and(x):
    return [Node("And", children=x)] if len(x) > 1 else x


def fix_between(inp):
    inp = re.sub(r"([\s|\S]+) >= (\d*) AND \1 <= (\d*)", r"\1 BETWEEN \2 and \3", inp)
    inp = re.sub(r"LIKE '([\s|\S]+?)'", r"LIKE '%\1%'", inp)
    return inp


class Item:
    def __init__(self, curr_type, l_child_idx, r_child_idx, mask):
        self.curr_type = curr_type
        self.l_child_idx = l_child_idx
        self.r_child_idx = r_child_idx
        self.mask = mask


class ZeroItem:
    def __init__(
        self,
        curr_type,
        final_leaf_indices,
        span_start_indices,
        span_end_indices,
        entities,
        enc,
        tokenizer,
    ):
        self.curr_type = curr_type
        self.final_leaf_indices = final_leaf_indices
        self.span_start_indices = span_start_indices
        self.span_end_indices = span_end_indices
        self.entities = entities
        self.enc = enc
        self.tokenizer = tokenizer


def reconstruct_tree(
    op_names, binary_op_count, batch_el, idx, items, cnt, num_schema_leafs
):
    type_data = int(items[cnt].curr_type[batch_el][idx])
    tuple_el = Node(op_names[type_data])
    if cnt > 0:
        if type_data < binary_op_count:
            l_idx = items[cnt].l_child_idx[batch_el][idx]
            r_idx = items[cnt].r_child_idx[batch_el][idx]

            l_child = reconstruct_tree(
                op_names,
                binary_op_count,
                batch_el,
                l_idx,
                items,
                cnt - 1,
                num_schema_leafs,
            )
            r_child = reconstruct_tree(
                op_names,
                binary_op_count,
                batch_el,
                r_idx,
                items,
                cnt - 1,
                num_schema_leafs,
            )
            tuple_el.children = [l_child, r_child]
        else:
            idx = items[cnt].l_child_idx[batch_el][idx]
            child = reconstruct_tree(
                op_names,
                binary_op_count,
                batch_el,
                idx,
                items,
                cnt - 1,
                num_schema_leafs,
            )
            tuple_el.children = [child]
    else:
        if idx < num_schema_leafs:
            entities = items[cnt].entities[batch_el]
            entity_idx = items[cnt].final_leaf_indices[batch_el][idx]
            tuple_el.val = entities[entity_idx]
        else:
            span_idx = idx - num_schema_leafs
            enc_tokens = items[cnt].enc["tokens"]["token_ids"][batch_el][1:].tolist()
            start_id = items[cnt].span_start_indices[batch_el][span_idx]
            end_id = items[cnt].span_end_indices[batch_el][span_idx]
            tuple_el.val = (
                items[cnt].tokenizer.decode(enc_tokens[start_id : end_id + 1]).strip()
            )
    return tuple_el


def remove_keep(node: Node):
    if node.name == "keep":
        node = remove_keep(node.children[0])
    node.children = [remove_keep(child) for child in node.children]
    return node


def promote(node, root=False):
    children = node.children
    if node.name in ["Having"]:
        while True:
            if not node.is_root and node.parent.name not in [
                "union",
                "intersect",
                "Subquery",
                "except",
            ]:
                prev_parent = node.parent
                grandparent = (
                    prev_parent.parent if not prev_parent.is_root else prev_parent
                )
                node.parent = grandparent
            else:
                break
        node.siblings[0].parent = node
    for child in children:
        promote(child)


def flatten_cnf(in_node):
    if in_node.name in ["And", "Or", "Val_list", "Product"]:
        return flatten_cnf_recurse(in_node, in_node.name, is_root=True)
    else:
        children_list = []
        for child in in_node.children:
            child.parent = None
            child = flatten_cnf(child)
            children_list.append(child)
        in_node.children = children_list
        return in_node


def flatten_cnf_recurse(in_node, n_type, is_root=False):
    other_op = "And" if n_type == "Or" else "Or"
    if in_node.name == n_type:
        res = []
        for child in in_node.children:
            child.parent = None
            res += flatten_cnf_recurse(child, n_type)
        if is_root:
            in_node.children = res
            return in_node
        else:
            return res
    elif in_node.name == other_op:
        return [flatten_cnf_recurse(in_node, other_op, True)]
    else:
        if not is_root:
            children_list = []
            for child in in_node.children:
                child.parent = None
                child = flatten_cnf(child)
                children_list.append(child)
            in_node.children = children_list
        return [in_node]


def irra_to_sql(tree, peren=True):
    if len(tree.children) == 0:
        if tree.name == "Table" and isinstance(tree.val, dict):
            return tree.val["value"] + " AS " + tree.val["name"]
        if hasattr(tree, "val"):
            return str(tree.val)
        else:
            print(tree)
            return ""
    if len(tree.children) == 1:
        if tree.name in [
            "min",
            "count",
            "max",
            "avg",
            "sum",
        ]:
            return "".join(
                [tree.name.upper(), "( ", irra_to_sql(tree.children[0]), " )"]
            )
        elif tree.name == "distinct":
            return "DISTINCT " + irra_to_sql(tree.children[0])
        elif tree.name == "literal":
            return """\'""" + str(irra_to_sql(tree.children[0])) + """\'"""
        elif tree.name == "Subquery":
            if peren:
                return "".join(["(", irra_to_sql(tree.children[0]), ")"])
            else:
                return irra_to_sql(tree.children[0])
        elif tree.name == "Join_on":
            tree = tree.children[0]
            if tree.name == "eq":
                first_table_name = tree.children[0].val.split(".")[0]
                second_table_name = tree.children[1].val.split(".")[0]
                return f"{first_table_name} JOIN {second_table_name} ON {tree.children[0].val} = {tree.children[1].val}"
            else:
                if len(tree.children) > 0:
                    t_Res = ", ".join([child.val for child in tree.children])
                    return t_Res
                else:
                    return tree.val
        else:  # Predicate or Table or 'literal' or Agg
            return irra_to_sql(tree.children[0])
    else:
        if tree.name in [
            "eq",
            "like",
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
        ]:
            pren_t = tree.name in [
                "eq",
                "like",
                "nin",
                "lte",
                "lt",
                "neq",
                "in",
                "gte",
                "gt",
            ]
            return (
                pred_dict[tree.name]
                .upper()
                .join([irra_to_sql(child, pren_t) for child in tree.children])
            )
        elif tree.name == "Orderby_desc":
            return (
                irra_to_sql(tree.children[1])
                + " ORDER BY "
                + irra_to_sql(tree.children[0])
                + " DESC"
            )
        elif tree.name == "Orderby_asc":
            return (
                irra_to_sql(tree.children[1])
                + " ORDER BY "
                + irra_to_sql(tree.children[0])
                + " ASC"
            )
        elif tree.name == "Project":
            return (
                "SELECT "
                + irra_to_sql(tree.children[0])
                + " FROM "
                + irra_to_sql(tree.children[1])
            )
        elif tree.name == "Join_on":
            # tree
            def table_name(x):
                return x.val.split(".")[0]

            table_tups = [
                (table_name(child.children[0]), table_name(child.children[1]))
                for child in tree.children
            ]
            res = table_tups[0][0]
            seen_tables = set(res)
            for (first, sec), child in zip(table_tups, tree.children):
                tab = first if sec in seen_tables else sec
                res += (
                    f" JOIN {tab} ON {child.children[0].val} = {child.children[1].val}"
                )
                seen_tables.add(tab)

            return res
        elif tree.name == "Selection":
            if len(tree.children) == 1:
                return irra_to_sql(tree.children[0])
            return (
                irra_to_sql(tree.children[1])
                + " WHERE "
                + irra_to_sql(tree.children[0])
            )
        else:  # 'Selection'/'Groupby'/'Limit'/Having'
            return (
                irra_to_sql(tree.children[1])
                + else_dict[tree.name]
                + irra_to_sql(tree.children[0])
            )


def ra_to_irra(tree):
    flat_tree = flatten_cnf(copy.deepcopy(tree))
    for node in findall(flat_tree, filter_=lambda x: x.name == "Selection"):
        table_node = node.children[1]
        join_list = []
        where_list = []
        having_list = []
        if node.children[0].name == "And":
            for predicate in node.children[0].children:
                if (
                    all(node_util.is_field(child) for child in predicate.children)
                    and predicate.name == "eq"
                ):
                    join_list.append(predicate)
                else:
                    if predicate.name == "Or" or all(
                        child.name in ["literal", "Subquery", "Value", "Or"]
                        for child in predicate.children
                    ):
                        where_list.append(predicate)
                    else:
                        having_list.append(predicate)
                predicate.parent = None
        else:
            if node.children[0].name == "eq" and all(
                node_util.is_field(child) for child in node.children[0].children
            ):
                join_list = [node.children[0]]
            elif node.children[0].name == "Or":
                where_list = [node.children[0]]
            else:
                if all(
                    child.name in ["literal", "Subquery", "Value", "Or"]
                    for child in node.children[0].children
                ):
                    where_list = [node.children[0]]
                else:
                    having_list = [node.children[0]]
            node.children[0].parent = None
        having_node = (
            [Node("Having", children=wrap_and(having_list))] if having_list else []
        )
        join_on = Node("Join_on", children=join_list)
        if len(join_on.children) == 0:
            join_on.children = [table_node]
        node.children = having_node + wrap_and(where_list) + [join_on]
    flat_tree = Node("Subquery", children=[flat_tree])
    promote(flat_tree)
    return flat_tree.children[0]


def ra_to_sql(tree):
    if tree:
        tree = remove_keep(tree)
        irra = ra_to_irra(tree)
        sql = irra_to_sql(irra)
        sql = fix_between(sql)
        sql = sql.replace("LIMIT value", "LIMIT 1")
        return sql
    else:
        return ""
