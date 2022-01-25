from pathlib import Path
import fiona
from hydamo_validation.utils import normalize_fiona_schema, read_geopackage
from hydamo_validation.summaries import ResultSummary
import numpy as np


class DataSets:
    """ValidationResult to be dumped as json."""

    def __init__(self, dataset_dir: str):
        """Initialie datasets."""
        self.path = Path(dataset_dir)
        self.properties = {}

        self._set_properties()

    def _set_properties(self):
        dataset_files = self.path.glob("*.gpkg")

        for gpkg in dataset_files:
            layers = fiona.listlayers(gpkg)
            layers_dict = {}
            for layer in layers:
                with fiona.open(gpkg, layer=layer) as src:
                    schema = normalize_fiona_schema(src.schema)
                    layers_dict[layer] = schema

            self.properties[gpkg.name] = layers_dict

    def _filter_status(self, gdf, status_object):
        if status_object is not None:
            if "statusobject" in gdf.columns:
                # gdf = gdf.loc[gdf["statusobject"].isin(status_object)]
                gdf = gdf.loc[
                    np.where(
                        gdf["statusobject"].isna()
                        | gdf["statusobject"].isin(status_object)
                    )
                ]
        return gdf

    @property
    def layers(self):
        """Return the layers in the datasets."""
        dataset_layers_dict = {k: list(v.keys()) for k, v in self.properties.items()}

        layers = [
            item for sublist in list(dataset_layers_dict.values()) for item in sublist
        ]

        return layers

    def read_layer(self, layer, result_summary=ResultSummary(), status_object=None):
        """
        Read a layer from the dataset.

        Parameters
        ----------
        layer : str
            Name of the layer (case sensitive!)
        result_summary : ResultSummary
            A hydamo_validation ResultSummary class where a possible exception
            will be appended to.
        status_object : List[str], optional
            A list of statusobject values used as a filter. The default is None.

        Raises
        ------
        e
            General exception while reading the layer from the geopackage.
        KeyError
            Specific exception; the layer is not part of the geopackage.

        Returns
        -------
        gdf : GeoDataFrame
            GeoDataFrame read from datasets (all columns are converted to lower case)
        schema : TYPE
            Fiona schema read from the layer
        """

        if layer in self.layers:
            dataset = {k: v for k, v in self.properties.items() if layer in v.keys()}
            file_path = self.path.joinpath(list(dataset.keys())[0])
            schema = list(dataset.values())[0][layer]
            try:
                gdf = read_geopackage(file_path, layer=layer)
                gdf = self._filter_status(gdf, status_object)
            except Exception as e:
                result_summary.append_warning(
                    (
                        f"Laag {layer} uit bestand {file_path.name} is geen "
                        "GeoPackage die wij kunnen openen. Vervang het bestand en "
                        "probeer opnieuw."
                    )
                )
                raise e

            # we will read all lower case
            schema["properties"] = {
                k.lower(): v for k, v in schema["properties"].items()
            }
            gdf.columns = [i.lower() for i in gdf.columns]
        else:
            raise KeyError(f"'{layer}' not in dataset-layers: '{self.layers}'")

        return gdf, schema
