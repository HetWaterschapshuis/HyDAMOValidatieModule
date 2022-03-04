"""Function to be picked up by the api."""

from typing import List, Callable, Literal
from pathlib import Path
import pandas as pd
from functools import partial
import json
import shutil
import logging
from jsonschema import validate, ValidationError
from hydamo_validation import logical_validation
from hydamo_validation.utils import Timer
from hydamo_validation.summaries import LayersSummary, ResultSummary
from hydamo_validation.datasets import DataSets
from hydamo_validation.datamodel import HyDAMO
from hydamo_validation.syntax_validation import (
    datamodel_layers,
    missing_layers,
    fields_syntax,
)

OUTPUT_TYPES = ["geopackage", "geojson", "csv"]
LOG_LEVELS = Literal["INFO", "DEBUG"]
INDEX = "nen3610id"
INCLUDE_COLUMNS = ["code"]


def _read_schema(version, schema_path=Path(__file__).parent.joinpath(r"./schemas")):
    schema_json = schema_path.joinpath(fr"rules/rules_{version}.json").resolve()
    with open(schema_json) as src:
        schema = json.load(src)
    return schema


def validator(
    output_types: List[str] = OUTPUT_TYPES,
    log_level: Literal["INFO", "DEBUG"] = "INFO",
    coverages: dict = {},
    schemas_path: Path = Path(__file__).parent.joinpath(r"./schemas"),
) -> Callable[[str], dict]:
    """

    Parameters
    ----------
    output_types : List[str], optional
        The types of output files that will be written. Options are
        ["geojson", "csv", "geopackage"]. By default all will be written
    log_level : Literal['INFO', 'DEBUG'], optional
        Level for logger. The default is "INFO".
    coverages : dict, optional
       Location of coverages. E.g. {"AHN: path_to_ahn_dir} The default is {}.
    schemas_path : Path, optional
        Path to the HyDAMO and validation_rules schemas.
        The default is Path(__file__).parent.joinpath(r"./schemas").

    Returns
    -------
    Callable[[str], dict]
        Partial of _validator function

    """

    return partial(
        _validator,
        output_types=output_types,
        log_level=log_level,
        coverages=coverages,
        schemas_path=schemas_path,
    )


def _validator(
    directory: str,
    output_types: List[str] = OUTPUT_TYPES,
    log_level: Literal["INFO", "DEBUG"] = "INFO",
    coverages: dict = {},
    schemas_path: Path = Path(__file__).parent.joinpath(r"./schemas"),
    raise_error: bool = False,
) -> dict:
    """

    Parameters
    ----------
    directory : str
        Directory with datasets sub-directory and validation_rules.json
    output_types : List[str], optional
        The types of output files that will be written. Options are
        ["geojson", "csv", "geopackage"]. By default all will be written
    log_level : Literal['INFO', 'DEBUG'], optional
        Level for logger. The default is "INFO".
    coverages : dict, optional
       Location of coverages. E.g. {"AHN: path_to_ahn_dir} The default is {}.
    schemas_path : Path, optional
        Path to the HyDAMO and validation_rules schemas.
        The default is Path(__file__).parent.joinpath(r"./schemas").
    raise_error: bool, optional
        Will raise an error (or not) when Exception is raised. The default is False

    Returns
    -------
    HyDAMO, LayersSummary, ResultSummary
        Will return a tuple with a filled HyDAMO datamodel, layers_summary and result_summary

    """
    timer = Timer()
    try:
        results_path = None
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level))
        date_check = pd.Timestamp.now().isoformat()
        result_summary = ResultSummary(date_check=date_check)
        layers_summary = LayersSummary(date_check=date_check)
        # check if all files are present
        dir_path = Path(directory)
        # create a results_path
        if dir_path.exists():
            results_path = dir_path.joinpath("results")
            if results_path.exists():
                try:
                    shutil.rmtree(results_path)
                except PermissionError:
                    pass
            results_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"{dir_path.absolute().resolve()} does not exist")

        dataset_path = dir_path.joinpath("datasets")
        validation_rules_json = dir_path.joinpath("validationrules.json")
        missing_paths = []
        for path in [dataset_path, validation_rules_json]:
            if not path.exists():
                missing_paths += [str(path)]
        if missing_paths:
            result_summary.error = f'missing_paths: {",".join(missing_paths)}'
            raise FileNotFoundError(f'missing_paths: {",".join(missing_paths)}')
        else:
            try:
                validation_rules_sets = json.loads(validation_rules_json.read_text())
            except Exception as e:
                result_summary.error = "the file with validationrules is not a valid JSON (see exception)"
                raise e
            try:
                rules_version = validation_rules_sets["schema"]
                schema = _read_schema(rules_version, schemas_path)
            except Exception as e:
                result_summary.error = "schema version cannot be read from validation rules (see exception)"
                raise e
            try:
                validate(validation_rules_sets, schema)
            except ValidationError as e:
                result_summary.error = (
                    f"validation rules invalid according to json-schema (see exception)"
                )
                raise e

        # check if output-files are supported
        unsupported_output_types = [
            item for item in output_types if item not in OUTPUT_TYPES
        ]
        if unsupported_output_types:
            error_message = (
                r"unsupported output types: " f'{",".join(unsupported_output_types)}'
            )
            result_summary.error = error_message
            raise TypeError(error_message)

        # set coverages
        if coverages:
            for key, item in coverages.items():
                logical_validation.general_functions._set_coverage(key, item)

        # start validation
        # read data-model
        result_summary.status = "define data-model"
        try:
            hydamo_version = validation_rules_sets["hydamo_version"]
            datamodel = HyDAMO(version=hydamo_version)
        except Exception as e:
            result_summary.error = "datamodel cannot be defined (see exception)"
            raise e

        # validate dataset syntax
        result_summary.status = "syntax-validation (layers)"
        datasets = DataSets(dataset_path)

        result_summary.dataset_layers = datasets.layers

        ## validate syntax of datasets on layers-level and append to result
        logger.info("syntax-validation of object-layers")
        valid_layers = datamodel_layers(datamodel.layers, datasets.layers)
        result_summary.missing_layers = missing_layers(
            datamodel.layers, datasets.layers
        )

        ## validate valid_layers on fields-level and add them to data_model
        result_summary.status = "syntax-validation (fields)"
        syntax_result = []

        ## get status_object if any
        status_object = None
        if "status_object" in validation_rules_sets.keys():
            status_object = validation_rules_sets["status_object"]

        for layer in valid_layers:
            logger.info(f"syntax-validation of fields in {layer}")
            gdf, schema = datasets.read_layer(
                layer, result_summary=result_summary, status_object=status_object
            )
            layer = layer.lower()
            for col in INCLUDE_COLUMNS:
                if not col in gdf.columns:
                    gdf[col] = None
                    schema["properties"][col] = "str"
            if not INDEX in gdf.columns:
                result_summary.error = f"Index-column '{INDEX}' is compulsory and not defined for layer '{layer}'."
                raise KeyError(f"{INDEX} not in columns")
            gdf, result_gdf = fields_syntax(
                gdf,
                schema,
                datamodel.validation_schemas[layer],
                INDEX,
                keep_columns=INCLUDE_COLUMNS,
            )

            # Add the syntax-validation result to the results_summary
            layers_summary.set_data(result_gdf, layer, schema["geometry"])
            # Add the corrected datasets_layer data to the datamodel.
            datamodel.set_data(gdf, layer, index_col=INDEX)
            syntax_result += [layer]

        # do logical validation: append result to layers_summary
        result_summary.status = "logical validation"
        layers_summary, result_summary = logical_validation.execute(
            datamodel,
            validation_rules_sets,
            layers_summary,
            result_summary,
            log_level,
            raise_error,
        )

        # finish validation and export results
        logger.info("exporting results")
        result_summary.status = "export results"
        result_layers = layers_summary.export(results_path, output_types)
        result_summary.result_layers = result_layers
        result_summary.error_layers = [
            i for i in datasets.layers if i.lower() not in result_layers
        ]
        result_summary.syntax_result = syntax_result
        result_summary.validation_result = [
            i["object"] for i in validation_rules_sets["objects"] if i["object"] in result_layers
        ]
        result_summary.success = True
        result_summary.status = "finished"
        result_summary.duration = timer.report()
        result_summary.to_json(results_path)
        logger.info(f"finished in {result_summary.duration:.2f} seconds")

        return datamodel, layers_summary, result_summary

    except Exception as e:
        e_str = str(e).replace("\n", " ")
        e_str = " ".join(e_str.split())
        if result_summary.error is not None:
            result_summary.error = fr"{result_summary.error} Python Exception: '{e_str}'"
        else:
            result_summary.error = fr"Python Exception: '{e_str}'"
        if results_path is not None:
            result_summary.to_json(results_path)
        if raise_error:
            raise e
        else:
            result_summary.to_dict()
