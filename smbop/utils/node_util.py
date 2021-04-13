import anytree
from itertools import *
from anytree import Node
from anytree.search import *


def is_number(x):
    """
    Takes a word and checks if Number (Integer or Float).
    """
    try:
        # only integers and float converts safely
        num = float(x)
        return True
    except:  # not convertable to float
        return False


def is_field(node):
    return hasattr(node, "val") and node.name == "Value" and not is_number(node.val)


def get_literals(tree):
    value_list = []

    def func(x):
        return hasattr(x, "val") and (
            isinstance(x.val, int) or isinstance(x.val, float)
        )

    for node in findall(tree, filter_=func):
        value_list.append(str(node.val))
    for node in findall(tree, filter_=lambda x: x.name == "literal"):
        value_list.append(node.children[0].val)
    return value_list


def print_tree(root, print_hash=True):
    tree = [
        f"{pre}{node.name} {node.val if hasattr(node, 'val') else ''} {node.hash if hasattr(node, 'hash') and print_hash  else ''}"
        for pre, fill, node in anytree.RenderTree(root)
    ]
    return "\n".join(tree)


def pad_with_keep(node, i):
    root = node
    prev = None
    for k in range(i):
        curr = Node("keep", parent=prev, max_depth=node.max_depth)
        if k == 0:
            root = curr
        prev = curr
    node.parent = prev
    return root


def add_max_depth_att(node):
    if not node.children:
        node.max_depth = node.depth
    else:
        node.children = [add_max_depth_att(child) for child in node.children]
        node.max_depth = max([child.max_depth for child in node.children])
    assert hasattr(node, "max_depth")
    return node


def tree2maxdepth(tree):
    if tree.parent:
        tree = pad_with_keep(tree, tree.parent.max_depth - tree.max_depth)
    if tree.children:
        tree.children = [tree2maxdepth(child) for child in tree.children]
    return tree


def get_leafs(tree):
    res = []
    for y in findall(tree, filter_=lambda x: hasattr(x, "val")):
        if isinstance(y.val, dict):
            if "value" in y.val:
                s = y.val["value"]
            elif "literal" in y.val:
                s = y.val["literal"]
            else:
                print(y.val)
                print(y)
                print(print_tree(tree))
                raise Exception
        elif isinstance(y.val, str):
            s = y.val
        elif isinstance(y.val, int) or isinstance(y.val, float):
            s = y.val  # TODO: fixme
        else:
            print(y.val)
            print(y)
            print(y.parent.name)
            raise Exception
        res.append(s)

    return res


RULES_novalues = """[["And", ["And", "like"]], ["And", ["eq", "gt"]], ["And", ["And", "neq"]], ["And", ["eq", "nin"]], ["And", ["eq", "gte"]], ["And", ["eq", "in"]], ["And", ["lt", "neq"]], ["And", ["eq", "eq"]], ["And", ["eq", "And"]], ["And", ["gt", "eq"]], ["And", ["gte", "gte"]], ["And", ["eq", "neq"]], ["And", ["And", "gte"]], ["And", ["like", "neq"]], ["And", ["gt", "lt"]], ["And", ["eq", "like"]], ["And", ["gt", "nin"]], ["And", ["gt", "lte"]], ["And", ["And", "gt"]], ["And", ["gt", "gt"]], ["And", ["eq", "Or"]], ["And", ["gte", "gt"]], ["And", ["eq", "lte"]], ["And", ["lt", "eq"]], ["And", ["gt", "in"]], ["And", ["eq", "lt"]], ["And", ["And", "Or"]], ["And", ["in", "in"]], ["And", ["gt", "gte"]], ["And", ["gte", "lte"]], ["And", ["gt", "neq"]], ["And", ["And", "And"]], ["And", ["And", "eq"]], ["And", ["gte", "eq"]], ["And", ["And", "lt"]], ["Groupby", ["Val_list", "Project"]], ["Groupby", ["Value", "Project"]], ["Limit", ["Value", "Orderby_desc"]], ["Limit", ["Value", "Orderby_asc"]], ["Or", ["neq", "neq"]], ["Or", ["gt", "gt"]], ["Or", ["eq", "lt"]], ["Or", ["like", "like"]], ["Or", ["gt", "eq"]], ["Or", ["eq", "eq"]], ["Or", ["gte", "gte"]], ["Or", ["lt", "gt"]], ["Or", ["eq", "gt"]], ["Or", ["gt", "lt"]], ["Orderby_asc", ["sum", "Groupby"]], ["Orderby_asc", ["Value", "Project"]], ["Orderby_asc", ["Value", "Groupby"]], ["Orderby_asc", ["Val_list", "Project"]], ["Orderby_asc", ["avg", "Groupby"]], ["Orderby_asc", ["count", "Groupby"]], ["Orderby_desc", ["avg", "Groupby"]], ["Orderby_desc", ["Value", "Project"]], ["Orderby_desc", ["count", "Groupby"]], ["Orderby_desc", ["sum", "Groupby"]], ["Orderby_desc", ["max", "Groupby"]], ["Orderby_desc", ["Value", "Groupby"]], ["Product", ["Product", "Table"]], ["Product", ["Table", "Table"]], ["Project", ["Val_list", "Product"]], ["Project", ["distinct", "Table"]], ["Project", ["min", "Table"]], ["Project", ["Val_list", "Table"]], ["Project", ["min", "Selection"]], ["Project", ["Val_list", "Selection"]], ["Project", ["count", "Selection"]], ["Project", ["sum", "Table"]], ["Project", ["avg", "Table"]], ["Project", ["Value", "Selection"]], ["Project", ["max", "Table"]], ["Project", ["max", "Selection"]], ["Project", ["Value", "Table"]], ["Project", ["count", "Table"]], ["Project", ["avg", "Selection"]], ["Project", ["sum", "Selection"]], ["Project", ["distinct", "Selection"]], ["Selection", ["And", "Product"]], ["Selection", ["Or", "Table"]], ["Selection", ["lte", "Table"]], ["Selection", ["neq", "Table"]], ["Selection", ["lt", "Product"]], ["Selection", ["gte", "Table"]], ["Selection", ["nin", "Table"]], ["Selection", ["eq", "Table"]], ["Selection", ["gt", "Table"]], ["Selection", ["lt", "Table"]], ["Selection", ["in", "Table"]], ["Selection", ["And", "Table"]], ["Selection", ["eq", "Product"]], ["Selection", ["like", "Table"]], ["Selection", ["nlike", "Table"]], ["Subquery", ["Limit"]], ["Subquery", ["Groupby"]], ["Subquery", ["except"]], ["Subquery", ["Project"]], ["Subquery", ["union"]], ["Subquery", ["intersect"]], ["Table", []], ["Table", ["Subquery"]], ["Val_list", ["max", "min"]], ["Val_list", ["sum", "sum"]], ["Val_list", ["Val_list", "avg"]], ["Val_list", ["Value", "Value"]], ["Val_list", ["Value", "sum"]], ["Val_list", ["avg", "min"]], ["Val_list", ["avg", "count"]], ["Val_list", ["Val_list", "max"]], ["Val_list", ["avg", "Value"]], ["Val_list", ["sum", "max"]], ["Val_list", ["avg", "sum"]], ["Val_list", ["count", "sum"]], ["Val_list", ["max", "max"]], ["Val_list", ["Value", "max"]], ["Val_list", ["sum", "avg"]], ["Val_list", ["max", "Value"]], ["Val_list", ["max", "sum"]], ["Val_list", ["Val_list", "count"]], ["Val_list", ["count", "max"]], ["Val_list", ["count", "Value"]], ["Val_list", ["distinct", "Value"]], ["Val_list", ["sum", "min"]], ["Val_list", ["min", "min"]], ["Val_list", ["count", "count"]], ["Val_list", ["count", "avg"]], ["Val_list", ["sum", "Value"]], ["Val_list", ["avg", "max"]], ["Val_list", ["min", "Value"]], ["Val_list", ["Val_list", "min"]], ["Val_list", ["Val_list", "sum"]], ["Val_list", ["min", "avg"]], ["Val_list", ["Value", "avg"]], ["Val_list", ["max", "avg"]], ["Val_list", ["Value", "count"]], ["Val_list", ["avg", "avg"]], ["Val_list", ["Val_list", "Value"]], ["Val_list", ["min", "max"]], ["Value", []], ["avg", ["Value"]], ["count", ["distinct"]], ["count", ["Value"]], ["distinct", ["Value"]], ["eq", ["Value", "Subquery"]], ["eq", ["Value", "Value"]], ["eq", ["Value", "literal"]], ["eq", ["count", "literal"]], ["except", ["Subquery", "Subquery"]], ["gt", ["avg", "Subquery"]], ["gt", ["count", "Subquery"]], ["gt", ["avg", "literal"]], ["gt", ["Value", "Subquery"]], ["gt", ["max", "literal"]], ["gt", ["Value", "literal"]], ["gt", ["Value", "Value"]], ["gt", ["count", "literal"]], ["gt", ["sum", "literal"]], ["gte", ["Value", "Subquery"]], ["gte", ["sum", "literal"]], ["gte", ["Value", "literal", "literal"]], ["gte", ["count", "literal"]], ["gte", ["Value", "literal"]], ["gte", ["Value", "Subquery", "literal"]], ["gte", ["avg", "literal"]], ["gte", ["count", "literal", "literal"]], ["in", ["Value", "Subquery"]], ["intersect", ["Subquery", "Subquery"]], ["like", ["Value", "literal"]], ["literal", ["Value"]], ["lt", ["Value", "Subquery"]], ["lt", ["min", "literal"]], ["lt", ["Value", "Value"]], ["lt", ["count", "literal"]], ["lt", ["avg", "literal"]], ["lt", ["Value", "literal"]], ["lte", ["Value"]], ["lte", ["count"]], ["lte", ["Value", "literal"]], ["lte", ["Value", "Subquery"]], ["lte", ["sum", "literal"]], ["lte", ["count", "literal"]], ["max", ["Value"]], ["min", ["Value"]], ["neq", ["Value", "literal"]], ["neq", ["Value", "Subquery"]], ["neq", ["Value", "Value"]], ["nin", ["Value", "Subquery"]], ["nlike", ["Value", "literal"]], ["sum", ["Value"]], ["union", ["Subquery", "Subquery"]]]"""
RULES_values = """[["Or", ["neq", "neq"]], ["Orderby_desc", ["max", "Groupby"]], ["Or", ["eq", "lt"]], ["And", ["lt", "neq"]], ["Selection", ["like", "Table"]], ["And", ["gte", "lte"]], ["And", ["eq", "And"]], ["Val_list", ["sum", "Value"]], ["Project", ["min", "Table"]], ["Or", ["lt", "gt"]], ["Selection", ["gte", "Table"]], ["Selection", ["lt", "Product"]], ["And", ["gte", "gte"]], ["lte", ["Value", "literal"]], ["Project", ["distinct", "Table"]], ["Subquery", ["intersect"]], ["And", ["And", "And"]], ["count", ["Value"]], ["Orderby_desc", ["Value", "Project"]], ["And", ["eq", "neq"]], ["Or", ["like", "like"]], ["Limit", ["Value", "Orderby_asc"]], ["gt", ["Value", "Subquery"]], ["Val_list", ["max", "max"]], ["Or", ["eq", "gt"]], ["Val_list", ["min", "min"]], ["Val_list", ["Val_list", "Value"]], ["sum", ["Value"]], ["Selection", ["eq", "Product"]], ["Project", ["sum", "Selection"]], ["Val_list", ["count", "Value"]], ["neq", ["Value", "literal"]], ["Orderby_asc", ["avg", "Groupby"]], ["Val_list", ["min", "Value"]], ["min", ["Value"]], ["Or", ["gt", "lt"]], ["eq", ["Value", "Subquery"]], ["lt", ["Value", "Subquery"]], ["Val_list", ["count", "max"]], ["Selection", ["And", "Product"]], ["gte", ["avg", "Value"]], ["Val_list", ["Val_list", "count"]], ["Project", ["Val_list", "Selection"]], ["lte", ["Value", "Value"]], ["Val_list", ["sum", "min"]], ["Or", ["gt", "gt"]], ["Val_list", ["max", "min"]], ["gt", ["count", "Value"]], ["Product", ["Table", "Table"]], ["neq", ["Value", "Value"]], ["And", ["lt", "eq"]], ["And", ["eq", "nin"]], ["Orderby_asc", ["Val_list", "Project"]], ["Groupby", ["Val_list", "Project"]], ["Val_list", ["Val_list", "min"]], ["gte", ["Value", "literal"]], ["gt", ["avg", "Value"]], ["eq", ["count", "Value"]], ["Project", ["avg", "Table"]], ["lt", ["count", "Value"]], ["Orderby_desc", ["avg", "Groupby"]], ["Val_list", ["count", "sum"]], ["And", ["eq", "eq"]], ["lt", ["min", "Value"]], ["Selection", ["Or", "Product"]], ["And", ["gt", "in"]], ["Or", ["gt", "eq"]], ["Val_list", ["sum", "avg"]], ["lt", ["avg", "Value"]], ["Project", ["max", "Selection"]], ["Val_list", ["sum", "sum"]], ["And", ["And", "lt"]], ["Limit", ["Value", "Orderby_desc"]], ["Selection", ["eq", "Table"]], ["gt", ["max", "Value"]], ["Orderby_asc", ["Value", "Groupby"]], ["Project", ["max", "Table"]], ["And", ["eq", "gt"]], ["literal", ["Value"]], ["Val_list", ["avg", "Value"]], ["gt", ["Value", "literal"]], ["gte", ["Value", "Value"]], ["Selection", ["lte", "Table"]], ["Selection", ["And", "Table"]], ["Project", ["count", "Selection"]], ["Val_list", ["Val_list", "sum"]], ["And", ["gte", "gt"]], ["And", ["gt", "lt"]], ["And", ["in", "in"]], ["Val_list", ["Value", "max"]], ["in", ["Value", "Subquery"]], ["lte", ["sum", "Value"]], ["Selection", ["neq", "Table"]], ["lt", ["Value", "literal"]], ["And", ["And", "lte"]], ["Val_list", ["avg", "count"]], ["Project", ["avg", "Selection"]], ["Val_list", ["Value", "count"]], ["Val_list", ["max", "Value"]], ["union", ["Subquery", "Subquery"]], ["Selection", ["gt", "Table"]], ["Val_list", ["sum", "max"]], ["except", ["Subquery", "Subquery"]], ["Subquery", ["Project"]], ["And", ["gt", "gt"]], ["Project", ["count", "Table"]], ["Val_list", ["Value", "avg"]], ["gt", ["Value", "Value"]], ["And", ["eq", "Or"]], ["Project", ["Value", "Table"]], ["like", ["Value", "literal"]], ["Orderby_desc", ["Value", "Groupby"]], ["And", ["gt", "lte"]], ["Val_list", ["distinct", "Value"]], ["Val_list", ["Value", "sum"]], ["Selection", ["lt", "Table"]], ["And", ["eq", "lt"]], ["And", ["gt", "gte"]], ["Orderby_asc", ["Value", "Project"]], ["Val_list", ["avg", "min"]], ["eq", ["Value", "Value"]], ["And", ["And", "Or"]], ["Val_list", ["avg", "max"]], ["Subquery", ["union"]], ["Orderby_asc", ["count", "Groupby"]], ["lt", ["Value", "Value"]], ["Subquery", ["Groupby"]], ["Project", ["Val_list", "Product"]], ["Val_list", ["min", "max"]], ["Selection", ["in", "Table"]], ["And", ["like", "neq"]], ["And", ["gte", "eq"]], ["count", ["distinct"]], ["Project", ["distinct", "Selection"]], ["lte", ["Value", "Subquery"]], ["Subquery", ["Limit"]], ["Or", ["gte", "gte"]], ["Val_list", ["Value", "Value"]], ["Orderby_asc", ["sum", "Groupby"]], ["And", ["eq", "lte"]], ["max", ["Value"]], ["Selection", ["nlike", "Table"]], ["Or", ["eq", "eq"]], ["gte", ["sum", "Value"]], ["And", ["eq", "gte"]], ["Product", ["Product", "Table"]], ["Val_list", ["min", "avg"]], ["eq", ["Value", "literal"]], ["nlike", ["Value", "literal"]], ["Selection", ["nin", "Table"]], ["Val_list", ["count", "count"]], ["neq", ["Value", "Subquery"]], ["Val_list", ["avg", "avg"]], ["gt", ["avg", "Subquery"]], ["Project", ["Value", "Selection"]], ["Val_list", ["avg", "sum"]], ["And", ["And", "gte"]], ["And", ["eq", "like"]], ["Orderby_desc", ["count", "Groupby"]], ["distinct", ["Value"]], ["gte", ["count", "Value"]], ["lte", ["count", "Value"]], ["And", ["And", "neq"]], ["And", ["And", "like"]], ["And", ["And", "eq"]], ["Val_list", ["Val_list", "max"]], ["gt", ["sum", "Value"]], ["Val_list", ["max", "avg"]], ["Orderby_desc", ["sum", "Groupby"]], ["Project", ["sum", "Table"]], ["Groupby", ["Value", "Project"]], ["Selection", ["Or", "Table"]], ["Val_list", ["max", "sum"]], ["Table", ["Subquery"]], ["avg", ["Value"]], ["intersect", ["Subquery", "Subquery"]], ["gte", ["Value", "Subquery"]], ["And", ["gt", "neq"]], ["nin", ["Value", "Subquery"]], ["Val_list", ["Val_list", "avg"]], ["And", ["gt", "eq"]], ["And", ["And", "gt"]], ["Project", ["Val_list", "Table"]], ["Val_list", ["count", "avg"]], ["Project", ["min", "Selection"]]]"""
