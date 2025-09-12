import sys
from .reader import TiledImageReader

def main() -> int:
    reader = TiledImageReader('./data/tiledimg.ome.tiff')
    print(reader.file_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
