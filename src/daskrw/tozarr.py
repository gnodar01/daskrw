import sys
import os
import pathlib
from .reader import TiledImageReader

def main() -> int:
    reader = TiledImageReader(pathlib.Path(os.path.dirname(__file__)) / 'data/tiledimg.ome.tiff')
    img = reader.read_tiled()

    return 0

if __name__ == "__main__":
    sys.exit(main())
