import os
import tempfile
from shutil import rmtree
from typing import TypedDict, Literal, Optional

import zarr
#import dask
#import numpy
from dask.array.core import Array as daskArray, to_zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata


class TiledImageWriter:
    """
    Writes tiled/pyramidal ome-zarr images
    """

    writer_name = "TiledImage"

    @staticmethod
    def create_temp_file(prefix="Cplargeimg", suffix=".ome.zarr"):
        tempdir = tempfile.gettempdir()
        if not (
            os.path.exists(tempdir) and os.access(tempdir, os.W_OK)
        ):
            tempdir = None

        fd, filepath = tempfile.mkstemp(
            prefix=prefix, suffix=suffix, dir=tempdir
        )

        os.close(fd)
        os.unlink(filepath)

        return filepath

    def __init_values(self):
        self.__zarr_location = None
        self.__zarr_store = None
        self.__zarr_root = None

    def __delete_state(self):
        del self.__zarr_location
        del self.__zarr_store
        del self.__zarr_root

    def __init__(self, file_path, img_shapes):
        """
        @param file_path: path to destination location, and filename
                          if None, a temp file is created automatically
        @param img_shapes: a list of size equal to the number of series,
                           with elements of size 5 of dimensions sizes for
                           t,c,z,y,x (in that order)
        """
        if file_path is None:
            file_path = self.create_temp_file()

        self.file_path = file_path
        self.__init_values()
        self.img_shapes = img_shapes

    def __del__(self):
        self.close()

    def __write_metadata(self, root):
        paths = [str(i) for i in range(len(self.img_shapes))]
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ]
        transformations = [ [{"type": "scale", "scale": [ 1.0, 1.0, pow(2, float(i)), pow(2, float(i)), pow(2, float(i)), ] }] for i in range(len(self.img_shapes)) ]
        datasets = []
        for p, t in zip(paths, transformations):
            datasets.append({"path": p, "coordinateTransformations": t})

        write_multiscales_metadata(root, datasets, axes=axes)

    def __get_root(self):
        if self.__zarr_root is None:
            self.__delete_state()
            self.__init_values()
            self.__zarr_location = parse_url(self.file_path, mode="w")
            assert self.__zarr_location is not None, "woops no zarr store"
            self.__zarr_store = self.__zarr_location.store
            self.__zarr_root = zarr.group(store=self.__zarr_store)
            self.__write_metadata(self.__zarr_root)
        return self.__zarr_root

    def write_tiled(self,
                    data: daskArray,
                    series=None,
                    c=None,
                    z=None,
                    t=None,
                    ):
        """Write a series of planes from the image file. Mimics the Bioformats API
        @param series: series (pyramid level)
        @param c: write from this channel. `None` = write color image if multichannel
            or interleaved RGB.
        @param z: z-stack index
        @param t: time index
        n.b. either z or t should be "None" to specify which channel to write across.
        """
        if series is None:
            series = "0"
        else:
            series = str(series)

        assert self.img_shapes is not None

        img_shape = self.img_shapes[int(series)]

        assert len(img_shape) == 5

        chunksize = (1,)*max(0, len(img_shape) - len(data.chunksize)) + data.chunksize

        root = self.__get_root()
        # get pyramid level, create if necessary
        zarray = root.require_dataset(
            series,
            shape=img_shape,
            exact=True,
            chunks=chunksize,
            dtype=data.dtype
        )

        idx_t = t or 0
        idx_c = c or 0
        idx_z = z or 0
        region = (
            slice(idx_t, idx_t+1, 1),
            slice(idx_c, idx_c+1, 1),
            slice(idx_z, idx_z+1, 1),
        )
        size_t = 1
        size_c = 1
        size_z = 1
        size_y = data.shape[0]
        size_x = data.shape[1]

        writedata = data.reshape(size_t, size_c, size_z, size_y, size_x)
        to_zarr(writedata, zarray, region=region)

        ## TODO: LIS - I want to be uisng map_blocks -> arr.store
        ## but write would need the x,y chunk index
        ## or better yet to_zarr with regions, but that doesn't work
        ## quite right for an unknown (to me) reason
        #for (iy, ix), chunk in numpy.ndenumerate(data.to_delayed()):
        #    arr = dask.compute(chunk)[0] # type: ignore
        #    y0 = iy * data.chunks[0][0]
        #    x0 = ix * data.chunks[1][0]
        #    y1 = y0 + arr.shape[0]
        #    x1 = x0 + arr.shape[1]
        #    zarray[t or 0, c or 0, z or 0, y0:y1, x0:x1] = arr

    def close(self):
        self.__delete_state()
        self.__init_values()

    def delete(self):
        if self.__zarr_root is not None:
            self.close()
        rmtree(self.file_path)

