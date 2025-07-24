"""Logical validation."""

# %%
from hydamo_validation import general_functions, logic_functions, topologic_functions
from shapely.geometry import LineString, Point, Polygon
import numpy as np


GEOTYPE_MAPPING = {LineString: "LineString", Point: "Point", Polygon: "Polygon"}
SUMMARY_COLUMNS = [
    "valid",
    "invalid",
    "invalid_critical",
    "invalid_non_critical",
    "ignored",
    "summary",
    "tags_assigned",
    "tags_invalid",
]
LIST_SEPARATOR = ";"
NOTNA_COL_IGNORE = ["related_parameter"]
EXCEPTION_COL = "nen3610id"


def _process_general_function(gdf, function, input_variables):
    return getattr(general_functions, function)(gdf, **input_variables)


def _process_logic_function(gdf, function, input_variables):
    return getattr(logic_functions, function)(gdf, **input_variables)


def _process_topologic_function(gdf, datamodel, function, input_variables):
    return getattr(topologic_functions, function)(gdf, datamodel, **input_variables)


def _notna_indices(gdf, input_variables):
    cols = []
    for k, v in input_variables.items():
        if k not in NOTNA_COL_IGNORE:
            if type(v) is not list:
                v = [v]
            cols += [val for val in v if val in gdf.columns]
    cols = list(set(cols))
    return [index for index, row in gdf[cols].iterrows() if not row.isnull().any()]


def _nan_message(nbr_indices, object_layer, rule_id, rule_type):
    return (
        f"{nbr_indices} objecten niet meegenomen in {rule_type} "
        f"object: '{object_layer}', id: '{rule_id}' wegens no_data in invoer."
    )


def _add_related_gdf(input_variables, datamodel, object_layer):
    related_object = input_variables["related_object"]
    related_gdf = getattr(datamodel, related_object).copy()
    if related_gdf.empty:
        raise Exception(f"Layer '{related_object}' is empty. Rule cannot be executed")

    code_relation = f"{object_layer}id"
    if code_relation not in related_gdf.columns:
        raise KeyError(
            f"'{code_relation}' not in columns of layer '{related_object}': {related_gdf.columns}"
        )
    if "related_parameter" in input_variables.keys():
        related_parameter = input_variables["related_parameter"]
        if related_parameter.startswith("geometry"):
            related_parameter = "geometry"
        if related_parameter not in related_gdf.columns:
            raise KeyError(
                f"'{related_parameter}' not in columns of layer '{related_object}': {related_gdf.columns}"
            )
    input_variables["code_relation"] = code_relation
    input_variables["related_gdf"] = related_gdf
    input_variables.pop("related_object")
    return input_variables


def _add_join_gdf(input_variables, datamodel):
    input_variables["join_gdf"] = getattr(
        datamodel, input_variables["join_object"]
    ).copy()
    return input_variables


def gdf_add_summary(
    gdf,
    variable,
    rule_id,
    penalty,
    error_message,
    critical,
    tags,
    tags_indices,
    separator=LIST_SEPARATOR,
):
    gdf.loc[gdf[variable] == False, "rating"] -= penalty
    gdf.loc[gdf[variable] == False, "summary"] += f"{error_message}{separator}"
    gdf.loc[gdf[variable] == False, "invalid"] += f"{rule_id}{separator}"
    gdf.loc[gdf[variable] == True, "valid"] += f"{rule_id}{separator}"
    gdf.loc[gdf[variable].isna(), "ignored"] += f"{rule_id}{separator}"
    if critical:
        gdf.loc[gdf[variable] == False, "invalid_critical"] += f"{rule_id}{separator}"
    else:
        gdf.loc[gdf[variable] == False, "invalid_non_critical"] += (
            f"{rule_id}{separator}"
        )
    if tags is not None:
        gdf.loc[tags_indices, ("tags_assigned")] += f"{tags}{separator}"
        gdf.loc[gdf[variable] == False, "tags_invalid"] += f"{tags}{separator}"
    return gdf


# %%
def execute(
    datamodel,
    validation_rules_sets,
    layers_summary,
    result_summary,
    logger=None,
    raise_error=False,
):
    """Execute the logical validation."""

    object_rules_sets = [
        i
        for i in validation_rules_sets["objects"]
        if i["object"] in datamodel.data_layers
    ]
    logger.info(
        rf"lagen met valide objecten en regels: {[i['object'] for i in object_rules_sets]}"
    )
    for object_rules in object_rules_sets:
        col_translation: dict = {}

        object_layer = object_rules["object"]
        logger.info(f"{object_layer}: start")
        object_gdf = getattr(datamodel, object_layer).copy()

        # add summary columns
        object_gdf["rating"] = 10
        for col in SUMMARY_COLUMNS:
            object_gdf[col] = ""

        # general rule section
        if "general_rules" in object_rules.keys():
            general_rules = object_rules["general_rules"]
            general_rules_sorted = sorted(general_rules, key=lambda k: k["id"])
            for rule in general_rules_sorted:
                logger.info(
                    f"{object_layer}: uitvoeren general-rule met id {rule['id']}"
                )
                try:
                    result_variable = rule["result_variable"]
                    result_variable_name = (
                        f"general_{rule['id']:03d}_{rule['result_variable']}"
                    )

                    # get function
                    function = next(iter(rule["function"]))
                    input_variables = rule["function"][function]

                    # remove all nan indices
                    indices = _notna_indices(object_gdf, input_variables)
                    dropped_indices = [
                        i
                        for i in object_gdf.index[object_gdf.index.notna()]
                        if i not in indices
                    ]

                    # add object_relation
                    if "related_object" in input_variables.keys():
                        input_variables = _add_related_gdf(
                            input_variables, datamodel, object_layer
                        )
                    elif "custom_function_name" in input_variables.keys():
                        input_variables["hydamo"] = datamodel
                    elif "join_object" in input_variables.keys():
                        input_variables = _add_join_gdf(input_variables, datamodel)

                    if dropped_indices:
                        result_summary.append_warning(
                            _nan_message(
                                len(dropped_indices),
                                object_layer,
                                rule["id"],
                                "general_rule",
                            )
                        )
                    if object_gdf.loc[indices].empty:
                        object_gdf[result_variable] = np.nan
                    else:
                        result = _process_general_function(
                            object_gdf.loc[indices], function, input_variables
                        )
                        object_gdf.loc[indices, result_variable] = result

                        getattr(datamodel, object_layer).loc[
                            indices, result_variable
                        ] = result

                    col_translation = {
                        **col_translation,
                        result_variable: result_variable_name,
                    }
                except Exception as e:
                    logger.error(
                        f"{object_layer}: general_rule {rule['id']} crashed width Exception {e}"
                    )
                    result_summary.append_error(
                        (
                            "general_rule niet uitgevoerd. Inspecteer de invoer voor deze regel: "
                            f"(object: '{object_layer}', id: '{rule['id']}', function: '{function}', "
                            f"input_variables: {input_variables}, Reason (Exception): {e})"
                        )
                    )
                    if raise_error:
                        raise e
                    else:
                        pass

        validation_rules = object_rules["validation_rules"]
        validation_rules = [
            i for i in validation_rules if ("active" not in i.keys()) | i["active"]
        ]
        validation_rules_sorted = sorted(validation_rules, key=lambda k: k["id"])
        # validation rules section
        for rule in validation_rules_sorted:
            try:
                rule_id = rule["id"]
                logger.info(
                    f"{object_layer}: uitvoeren validatieregel met id {rule_id} ({rule['name']})"
                )
                result_variable = rule["result_variable"]
                if "exceptions" in rule.keys():
                    exceptions = rule["exceptions"]
                    indices = object_gdf.loc[
                        ~object_gdf[EXCEPTION_COL].isin(exceptions)
                    ].index
                else:
                    indices = object_gdf.index
                    exceptions = []
                result_variable_name = (
                    f"validate_{rule_id:03d}_{rule['result_variable']}"
                )

                # get function
                function = next(iter(rule["function"]))
                input_variables = rule["function"][function]

                # remove all nan indices
                notna_indices = _notna_indices(object_gdf, input_variables)
                indices = [i for i in indices[indices.notna()] if i in notna_indices]

                # add object_relation
                if "join_object" in input_variables.keys():
                    input_variables = _add_join_gdf(input_variables, datamodel)

                # apply filter on indices
                if "filter" in rule.keys():
                    filter_function = next(iter(rule["filter"]))
                    filter_input_variables = rule["filter"][filter_function]
                    series = _process_logic_function(
                        object_gdf, filter_function, filter_input_variables
                    )
                    series = series[series.index.notna()]
                    filter_indices = series.loc[series].index.to_list()
                    indices = [i for i in filter_indices if i in indices]
                else:
                    filter_indices = []

                if object_gdf.loc[indices].empty:
                    object_gdf[result_variable] = None
                elif rule["type"] == "logic":
                    object_gdf.loc[indices, (result_variable)] = (
                        _process_logic_function(
                            object_gdf.loc[indices], function, input_variables
                        )
                    )
                elif (rule["type"] == "topologic") and (
                    hasattr(datamodel, "hydroobject")
                ):
                    result_series = _process_topologic_function(
                        # getattr(
                        #     datamodel, object_layer
                        # ),  # FIXME: commented as we need to apply filter in topologic functions as well. Remove after tests pass
                        object_gdf,
                        datamodel,
                        function,
                        input_variables,
                    )
                    object_gdf.loc[indices, (result_variable)] = result_series.loc[
                        indices
                    ]

                col_translation = {
                    **col_translation,
                    result_variable: result_variable_name,
                }

                # summarize
                if rule["error_type"] == "critical":
                    penalty = 5
                    critical = True
                else:
                    penalty = 1
                    critical = False
                if "penalty" in rule.keys():
                    penalty = rule["penalty"]

                error_message = rule["error_message"]

                if "tags" in rule.keys():
                    tags = LIST_SEPARATOR.join(rule["tags"])
                else:
                    tags = None

                exceptions += filter_indices
                _valid_indices = object_gdf[~object_gdf.index.isna()].index
                tags_indices = [i for i in _valid_indices if i not in exceptions]
                object_gdf = gdf_add_summary(
                    gdf=object_gdf,
                    variable=result_variable,
                    rule_id=rule_id,
                    penalty=penalty,
                    error_message=error_message,
                    critical=critical,
                    tags=tags,
                    tags_indices=tags_indices,
                )

            except Exception as e:
                logger.error(
                    f"{object_layer}: validation_rule {rule['id']} width Exception {e}"
                )
                result_summary.append_error(
                    (
                        "validation_rule niet uitgevoerd. Inspecteer de invoer voor deze regel: "
                        f"(object '{object_layer}', rule_id '{rule['id']}', function: '{function}', "
                        f"input_variables: {input_variables}, Reason (Exception): {e})"
                    )
                )
                if raise_error:
                    raise e
                else:
                    pass

        # drop columns
        drop_columns = [
            i
            for i in object_gdf.columns
            if i
            not in list(col_translation.keys())
            + ["nen3610id", "geometry", "rating"]
            + SUMMARY_COLUMNS
        ]
        object_gdf.drop(columns=drop_columns, inplace=True)
        # re_order columns
        column_order = ["nen3610id"]
        column_order += list(col_translation.keys())
        column_order += ["rating"] + SUMMARY_COLUMNS
        if "geometry" in object_gdf.columns:
            column_order += ["geometry"]
        object_gdf = object_gdf[column_order]

        # finish result columns
        for i in SUMMARY_COLUMNS:
            if i in object_gdf.columns:
                object_gdf.loc[:, i] = object_gdf[i].map(lambda x: str(x)[:-1])
        if "rating" in object_gdf.columns:
            object_gdf.loc[:, "rating"] = np.maximum(1, object_gdf["rating"])
        for i in ["tags_assigned", "tags_invalid"]:
            if i in object_gdf.columns:
                object_gdf.loc[:, i] = object_gdf[i].map(
                    lambda x: ";".join(list(set(str(x).split(LIST_SEPARATOR))))
                )

        # rename columns
        object_gdf.rename(columns=col_translation, inplace=True)

        # join gdf to layer_summary
        layers_summary.join_gdf(object_gdf, object_layer)

    return layers_summary, result_summary
