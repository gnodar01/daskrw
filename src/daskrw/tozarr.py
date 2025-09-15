import sys
import os
import pathlib
from .reader import TiledImageReader
from .writer import TiledImageWriter

def main() -> int:
    reader = TiledImageReader(pathlib.Path(os.path.dirname(__file__)) / 'data/tiledimg.ome.tiff')
    # dtype: uint8
    # chunskize: 256, 256, 1
    # num levels: (series) 2
    #   level 0 shape: 512, 512, 2
    high_res_ch0 = reader.read_tiled(series=0, c=0)
    high_res_ch1 = reader.read_tiled(series=0, c=1)
    #   level 1 shape: 256, 256, 2
    low_res_ch0 = reader.read_tiled(series=1, c=0)
    low_res_ch1 = reader.read_tiled(series=1, c=1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
