import sys
import os
import pathlib

from dask.array.core import Array as daskArray

from .reader import TiledImageReader
from .writer import TiledImageWriter
from .constants import (
    MD_SIZE_S,
    MD_SIZE_C,
    MD_SIZE_Z,
    MD_SIZE_T,
    MD_SIZE_Y,
    MD_SIZE_X,
)

def main() -> int:
    reader = TiledImageReader(pathlib.Path(os.path.dirname(__file__)) / 'data/tiledimg.ome.tiff')

    metadata = reader.get_series_metadata()
    # [
    #     ( 1, 2, 1, 512, 512 ),
    #     ( 1, 2, 1, 256, 256 ),
    # ]
    img_shapes = list(zip(
        metadata[MD_SIZE_T],
        metadata[MD_SIZE_C],
        metadata[MD_SIZE_Z],
        metadata[MD_SIZE_Y],
        metadata[MD_SIZE_X],
    ))
    assert len(img_shapes) == metadata[MD_SIZE_S]

    # dtype: uint8
    # chunskize: 256, 256, 1
    # num levels: (series) 2
    #   level 0 shape: 512, 512, 2
    high_res_ch0: daskArray = reader.read_tiled(series=0, c=0) # type: ignore
    high_res_ch1: daskArray = reader.read_tiled(series=0, c=1) # type: ignore
    #   level 1 shape: 256, 256, 2
    low_res_ch0: daskArray = reader.read_tiled(series=1, c=0) # type: ignore
    low_res_ch1: daskArray = reader.read_tiled(series=1, c=1) # type: ignore


    writer = TiledImageWriter(None, img_shapes)
    print(f"Writing to temp file: {writer.file_path}")

    writer.write_tiled(high_res_ch0, series=0, c=0)
    writer.write_tiled(high_res_ch1, series=0, c=1)
    writer.write_tiled(low_res_ch0, series=1, c=0)
    writer.write_tiled(low_res_ch1, series=1, c=1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
