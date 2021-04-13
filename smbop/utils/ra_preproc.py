from functools import reduce
from itertools import *
from anytree import Node
from anytree.search import *


def createTable(x):
    return {"table": x}


def reduce_and(l):
    return reduce(lambda a, b: {"and": [a, b]}, l)


def reduce_or(l):
    return reduce(lambda a, b: {"or": [a, b]}, l)


def codegen_table(ast_dict, args):
    if ast_dict.get("product"):
        parent = Node("Product", n_type="Table")
        codegen_table(ast_dict["product"][0], args).parent = parent
        codegen_table(ast_dict["product"][1], args).parent = parent
        return parent
    else:
        parent = Node("Table", n_type="Table")
        if (
            isinstance(ast_dict["table"], dict)
            and isinstance(ast_dict["table"].get("value"), dict)
            and ast_dict["table"]["value"].get("query")
        ):
            res = ast_to_ra(ast_dict["table"]["value"]["query"], args)
            res.parent = Node("Subquery", parent=parent, n_type="Table")
        else:
            res = ast_dict["table"]
            parent.val = res

        return parent


def codegen_cnf(ast_dict):
    if isinstance(ast_dict, dict):
        if ast_dict.get("and"):
            # TODO: add parameter to change into list
            return reduce_and([codegen_cnf(el) for el in ast_dict["and"]])
        elif ast_dict.get("or"):
            return reduce_or([codegen_cnf(el) for el in ast_dict["or"]])
        return ast_dict
    else:
        return ast_dict


def codegen_from(inp, args):
    tables = []
    on_list = []
    if isinstance(inp, str) or isinstance(inp, dict):
        tables, on_list = [createTable(inp)], []
    else:

        for i in inp:
            if isinstance(i, dict):
                if i.get("join"):
                    tables.append(createTable(i.get("join")))
                    if not i.get("on"):
                        pass
                    elif i.get("on").get("and"):
                        on_list += i.get("on").get("and")
                    else:
                        on_list.append(i.get("on"))

                else:
                    tables.append(createTable(i))

            else:
                tables.append(createTable(i))
    tables = codegen_table(reduce(lambda a, b: {"product": [a, b]}, tables), args)
    return tables, on_list


def codegen_vallist(ast_dict):
    if ast_dict.get("val_list"):
        parent = Node("Val_list", n_type="Value")
        codegen_vallist(ast_dict["val_list"][0]).parent = parent
        codegen_vallist(ast_dict["val_list"][1]).parent = parent
        return parent
    else:
        return codegen_agg(ast_dict["value"])


def codegen_select(ast_dict, args):
    if isinstance(ast_dict, str) and ast_dict == "*":
        return Node("Value", val="*", n_type="Value")
    if isinstance(ast_dict, dict):
        return codegen_agg(ast_dict["value"])

    if isinstance(ast_dict, list):
        # TODO: add parameter to change into list
        ast_dict = reduce(lambda a, b: {"val_list": [a, b]}, ast_dict)
        return codegen_vallist(ast_dict)


def codegen_where(where_list, on_list, having_list, args):
    if on_list:
        on_list = reduce_and(on_list)
    where_list = codegen_cnf(where_list)
    having_list = codegen_cnf(having_list)
    res = []

    res += [on_list] if on_list else []
    res += [having_list] if having_list else []
    res += [where_list] if where_list else []

    if res:
        # TODO: add parameter to change into list
        res = reduce_and(res)
        res = codegen_cnf(res)
        res = codegen_predicate(res, args)
    return res


def codegen_predicate(ast_dict, args):
    if ast_dict.get("and"):
        parent = Node("And", n_type="Predicate")
        codegen_predicate(ast_dict["and"][0], args).parent = parent
        codegen_predicate(ast_dict["and"][1], args).parent = parent
        return parent
    elif ast_dict.get("or"):
        parent = Node("Or", n_type="Predicate")
        codegen_predicate(ast_dict["or"][0], args).parent = parent
        codegen_predicate(ast_dict["or"][1], args).parent = parent
        return parent
    predicate_type = list(ast_dict.keys())[0]
    predicate_node = Node(predicate_type, n_type="Predicate")
    if len(ast_dict[predicate_type]) == 2:
        val1, val2 = ast_dict[predicate_type]
        codegen_subquery(val1, args).parent = predicate_node
        codegen_subquery(val2, args).parent = predicate_node
    else:
        assert predicate_type == "between"
        val0, val1, val2 = ast_dict[predicate_type]
        predicate_node = Node("And", n_type="Predicate")
        pred1 = Node("gte", parent=predicate_node, n_type="Predicate")
        pred2 = Node("lte", parent=predicate_node, n_type="Predicate")
        codegen_agg(val0).parent = pred1
        codegen_subquery(val1, args).parent = pred1
        codegen_agg(val0).parent = pred2
        codegen_subquery(val2, args).parent = pred2
    return predicate_node


def codegen_subquery(ast_dict, args):
    if isinstance(ast_dict, dict) and ast_dict.get("query"):
        query = ast_to_ra(ast_dict["query"], args)
        curr = Node("Subquery", n_type="Table")
        query.parent = curr
        return curr
    else:
        return codegen_agg(ast_dict)


def codegen_agg(ast_dict):
    node = Node("Value", n_type="Value")
    if (
        isinstance(ast_dict, str)
        or isinstance(ast_dict, int)
        or isinstance(ast_dict, float)
    ):
        node.val = ast_dict
        return node
    if isinstance(ast_dict, dict):
        agg_type = list(ast_dict.keys())[0]
        agg_type_node = Node(agg_type, n_type="Agg")

        if isinstance(ast_dict[agg_type], dict):
            sec_agg_type = list(ast_dict[agg_type].keys())[0]
            if sec_agg_type in ["distinct"]:
                distinct_type_node = Node(
                    "distinct", parent=agg_type_node, n_type="Agg"
                )
                node.val = ast_dict[agg_type]["distinct"]
                node.parent = distinct_type_node
            elif sec_agg_type in ["add", "sub", "div", "mul"]:
                sec_agg_node = Node(sec_agg_type, parent=agg_type_node, n_type="Agg")
                val1, val2 = ast_dict[agg_type][sec_agg_type]
                codegen_agg(val1).parent = sec_agg_node
                codegen_agg(val2).parent = sec_agg_node
            else:
                raise Exception
        else:
            node.val = ast_dict[agg_type]
            node.parent = agg_type_node

        return agg_type_node
    else:
        print(ast_dict)


def ast_to_ra(ast_dict, args=None):
    if ast_dict.get("op"):
        res1 = ast_to_ra(ast_dict["op"]["query1"], args)

        res2 = ast_to_ra(ast_dict["op"]["query2"], args)
        c = Node(ast_dict["op"].get("type"), n_type="Op")
        parent1 = Node("Subquery", parent=c, n_type="Table")
        parent2 = Node("Subquery", parent=c, n_type="Table")
        res1.parent = parent1
        res2.parent = parent2
        return c

    root = Node("Project", n_type="Table")
    select_node = codegen_select(ast_dict["select"], args)
    select_node.parent = root
    node = root

    tables, on_list = codegen_from(ast_dict["from"], args)
    where_list = ast_dict.get("where")
    having_list = ast_dict.get("having")
    condition = codegen_where(where_list, on_list, having_list, args)
    if condition:
        node = Node("Selection", parent=node, n_type="Table")
        condition.parent = node

    tables.parent = node
    node = tables
    if ast_dict.get("groupby"):
        curr = Node("Groupby", n_type="Table")
        codegen_select(ast_dict["groupby"], args).parent = curr
        root.parent = curr
        root = root.parent
    if ast_dict.get("orderby"):
        if isinstance(ast_dict["orderby"], dict) and ast_dict["orderby"].get("sort"):
            sort = "Orderby_" + ast_dict["orderby"]["sort"]
        else:
            sort = "Orderby_asc"
        curr = Node(sort, n_type="Table")
        codegen_select(ast_dict["orderby"], args).parent = curr
        root.parent = curr
        root = root.parent

    if ast_dict.get("limit"):
        curr = Node("Limit", n_type="Table")
        val = ast_dict["limit"]
        if isinstance(val, dict):
            val = val["literal"]
        Node("Value", val=val, n_type="Value").parent = curr
        root.parent = curr
        root = root.parent
    return root
