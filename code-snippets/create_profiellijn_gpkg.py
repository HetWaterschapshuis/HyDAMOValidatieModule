""" "
Basis-code voor het genereren van een profiellijn.gpkg op basis van een profielpunt.gpkg

Let op (!) de volgende uitgangspunten:
- de kolom profiellijnid in profielpunt.gpkg bevat de code van het profiel (kan ook globalid zijn)
- begin- en eindcoordinaat worden bepaald op basis van codevolgnummer in profielpunt.
  Als deze niet klopt, zal de lijn ook niet correct worden getekend
- nen310id moet eigenlijk de waterbeheercode bevatten, we stoppen er nu de objectid in

"""

import geopandas as gpd
import shapely
from shapely.geometry import LineString
from pathlib import Path


# Zet juiste folder met datasets en juiste bestand
data_dir = Path("pad/naar/mijn/datasets")
profielpunt_gpkg = data_dir.joinpath(r"profielpunt.gpkg")
profiellijn_gpkg = data_dir.joinpath(r"profiellijn.gpkg")

# inlezen profielpunt-bestand
profielpunt_df = gpd.read_file(profielpunt_gpkg, layer="profielpunt")

# Groeperen punten naar profiellijnid en genereren profiellijn
data = []
for idx, (code, df) in enumerate(profielpunt_df.groupby("profiellijnid")):
    geometry = LineString(
        [
            df.set_index("codevolgnummer").sort_index().geometry.iloc[0],
            df.set_index("codevolgnummer").sort_index().geometry.iloc[-1],
        ]
    )

    data += [
        {
            "globalid": code,
            "code": code,
            "objectid": idx + 1,
            "nen3610id": str(idx + 1),
            "geometry": geometry,
        }
    ]

# definieren profiellijn en forceren naar 2D punt
profiellijn_df = gpd.GeoDataFrame(data, crs=profielpunt_df.crs)

profiellijn_df.loc[:, "geometry"] = gpd.GeoSeries(
    shapely.force_2d(profiellijn_df.geometry.array), crs=df.crs
)

# wegschrijven profiellijn
profiellijn_df.to_file(profiellijn_gpkg, layer="profiellijn")
