import json


class Replacer:
    def __init__(self, table_path) -> None:
        self.table_path = table_path
        self.mapping = {}
        with open(self.table_path) as f:
            schema_dicts = json.load(f)
            for schema_dict in schema_dicts:
                schema_dict["table_names"] = [
                    x.replace(" ", "_") for x in schema_dict["table_names"]
                ]
                schema_dict["table_names_original"] = [
                    x.replace(" ", "_") for x in schema_dict["table_names_original"]
                ]
                names = [
                    f"{schema_dict['table_names'][x[0]]}.{x[1]}".replace(" ", "_")
                    for x in schema_dict["column_names"]
                    if x[0] > -1
                ] + schema_dict["table_names"]
                names = [x.lower() for x in names]
                names_orig = [
                    f"{schema_dict['table_names_original'][x[0]]}.{x[1]}".replace(
                        " ", "_"
                    )
                    for x in schema_dict["column_names_original"]
                    if x[0] > -1
                ] + schema_dict["table_names_original"]
                names_orig = [x.lower() for x in names_orig]
                name2orig = {x: y for x, y in zip(names, names_orig)}
                orig2name = {y: x for x, y in zip(names, names_orig)}
                self.mapping[schema_dict["db_id"]] = {
                    "orig2name": orig2name,
                    "name2orig": name2orig,
                }

    def pre(self, str_in, db_id):
        if (
            isinstance(str_in, str)
            and str_in.lower() in self.mapping[db_id]["orig2name"]
        ):
            str_in = self.mapping[db_id]["orig2name"][str_in]
        return str_in

    def post(self, str_in, db_id):
        if (
            isinstance(str_in, str)
            and str_in.lower() in self.mapping[db_id]["name2orig"]
        ):
            str_in = self.mapping[db_id]["name2orig"][str_in]
        return str_in
