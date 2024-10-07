# %%
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, LineString

hydamo_gpkg = Path(__file__).parent.joinpath("datasets", "HyDAMO.gpkg")

# profielpunten
df = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries(
        [
            Point(0, 0, 0),
            Point(1, 0),
            LineString(((2, 0), (2, 1))),
            LineString(((3, 0, 0), (3, 1, 0))),
        ]
    ),
    crs=28992,
)

df.to_file(hydamo_gpkg, layer="Profielpunt")

# %%
