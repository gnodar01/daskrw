import os
import tempfile
from shutil import rmtree
from typing import TypedDict, Literal, Optional, Union

import zarr
import dask
import numpy
from dask.array.core import Array as daskArray
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscales_metadata


class WriteTracker(TypedDict):
    # pyramid level
    level: Optional[int]
    # tile number
    nth: Optional[int]
    # following may contain a single single idx, or a list of idxs
    c: Optional[Union[int, list[int]]]
    z: Optional[Union[int, list[int]]]
    t: Optional[Union[int, list[int]]]


class Resolution(TypedDict):
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    height: int
    width: int
    channels: int
    max_tile_height: int
    max_tile_width: int
    n_tiles_y: int
    n_tiles_x: int


class StandardMetadata(TypedDict):
    endiness: Literal["<", ">"]
    dim_order: str
    y_idx: int
    x_idx: int
    c_idx: int
    z_idx: int
    t_idx: int
    y_size: int
    x_size: int
    z_size: int
    c_size: int
    t_size: int
    y_mag: float
    y_mag_unit: str
    x_mag: float
    x_mag_unit: str
    z_mag: float
    z_mag_unit: str
    shape: tuple[int, ...]
    dtype: str
    channel_names: tuple[str, ...]
    tile_height: int
    tile_width: int
    resolutions: dict[int, Resolution]



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
        self._write_tracker: WriteTracker = {
            "level": None,
            "nth": None,
            "c": None,
            "z": None,
            "t": None,
        }
        self.__zarr_location = None
        self.__zarr_store = None
        self.__zarr_root = None

        # self.__cached_meta = None
        # self.__cached_full_meta = None
        # self.__dim_idxs = {
        #     "channel_idx": 0,
        #     "row_idx": 1,
        #     "col_idx": 2,
        # }

    def __delete_state(self):
        del self.__zarr_location
        del self.__zarr_store
        del self.__zarr_root

        del self.level
        del self.nth
        del self.channel
        del self.plane


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

        # TODO: LIS - I want to be uisng map_blocks -> arr.store
        # but write would need the x,y chunk index
        # or better yet to_zarr with regions, but that doesn't work
        # quite right for an unknown (to me) reason
        for (iy, ix), chunk in numpy.ndenumerate(data.to_delayed()):
            arr = dask.compute(chunk)[0] # type: ignore
            y0 = iy * data.chunks[0][0]
            x0 = ix * data.chunks[1][0]
            y1 = y0 + arr.shape[0]
            x1 = x0 + arr.shape[1]
            zarray[t or 0, c or 0, z or 0, y0:y1, x0:x1] = arr

    def close(self):
        self.__delete_state()
        self.__init_values()

    def delete(self):
        if self.__zarr_root is not None:
            self.close()
        rmtree(self.file_path)

    def get_level(self):
        return self._write_tracker["level"]

    def set_level(self, level: int):
        self._write_tracker["level"] = level

    def del_level(self):
        self._write_tracker["level"] = None

    level = property(get_level, set_level, del_level, "pyramid level")

    def get_nth(self):
        return self._write_tracker["nth"]

    def set_nth(self, nth: int):
        self._write_tracker["nth"] = nth

    def del_nth(self):
        self._write_tracker["nth"] = None

    nth = property(get_nth, set_nth, del_nth, "tile number")

    def get_channel(self):
        return self._write_tracker["c"]

    def set_channel(self, channel: Union[int, list[int]]):
        self._write_tracker["c"] = channel

    def del_channel(self):
        self._write_tracker["c"] = None

    channel = property(get_channel, set_channel, del_channel, "channel number(s)")

    def get_plane(self):
        return self._write_tracker["z"]

    def set_plane(self, plane: Union[int, list[int]]):
        self._write_tracker["z"] = plane

    def del_plane(self):
        self._write_tracker["z"] = None

    plane = property(get_plane, set_plane, del_plane, "plane number(s)")

    def get_frame(self):
        return self._write_tracker["t"]

    def set_frame(self, frame: Union[int, list[int]]):
        self._write_tracker["t"] = frame

    def del_frame(self):
        self._write_tracker["t"] = None

    frame = property(get_frame, set_frame, del_frame, "time frame number(s)")

    # def current_tile(self, all_channels=False):
    #     nth = self.nth
    #     level = self.level
    #     if all_channels:
    #         num_channels = self._res[level]["channels"]
    #         channel = slice(0,num_channels,1)
    #     else:
    #         channel = self.channel

    #     assert nth >= 0
    #     assert nth <= self._nn(level), f"only {self._nn(level)} tiles at level {level}, got {nth}"

    #     return self._tile_n(nth=nth, channel=channel, level=level)

    # def go_tile_left(self):
    #     nth = self.nth
    #     level = self.level
    #     curr_x = nth % self._nx(level)
    #     if curr_x > 0:
    #         self.nth = nth - 1
    #     return self.current_tile()

    # def go_tile_right(self):
    #     nth = self.nth
    #     level = self.level
    #     curr_x = nth % self._nx(level)
    #     if curr_x < (self._nx(level) - 1):
    #         self.nth = nth + 1
    #     return self.current_tile()

    # def go_tile_up(self):
    #     nth = self.nth
    #     level = self.level
    #     new_nth = nth - self._nx(level)
    #     if new_nth >= 0:
    #         self.nth = new_nth
    #     return self.current_tile()

    # def go_tile_down(self):
    #     nth = self.nth
    #     level = self.level
    #     new_nth = nth + self._nx(level)
    #     if new_nth < self._nn(level):
    #         self.nth = new_nth
    #     return self.current_tile()

    # #  down the inverted pyramid (downscale)
    # def go_level_up(self):
    #     level = self.level
    #     nth = self.nth
    #     if level < (len(self._res) - 1):
    #         new_iy = self._iy(level, nth) // 2
    #         new_ix = self._ix(level, nth) // 2

    #         level += 1

    #         new_nx = self._nx(level)

    #         self.level = level
    #         self.nth = new_iy * new_nx + new_ix
    #     return self.current_tile()

    # # up the inverted pyramid (upscale)
    # def go_level_down(self):
    #     level = self.level
    #     nth = self.nth
    #     if level > 0:
    #         new_iy = self._iy(level, nth) * 2
    #         new_ix = self._ix(level, nth) * 2

    #         level = max(0, level - 1)

    #         new_nx = self._nx(level)

    #         self.level = level
    #         self.nth = new_iy * new_nx + new_ix
    #     return self.current_tile()

    # def _tile_n(self, nth: int, channel: slice = slice(0,1,1), level: int = 0) -> daskArray:
    #     assert self.__data, "No data read yet (read_tile failed or was never called)"
    #     assert len(self.__data) > level
    #     assert level >= 0

    #     _res = self._res
    #     row_slice, col_slice = self._n_slices(nth, level)

    #     tile = self.__data[level][row_slice, col_slice, channel]

    #     assert 0 not in tile.shape, f"invalid shape {tile.shape}"

    #     return tile

    # def _decrement_channel(self, curr_channel: slice, level: Optional[int]) -> slice:
    #     return self._set_channel(start = curr_channel.start - 1, stop = curr_channel.stop - 1, step = curr_channel.step, level = level)

    # def _increment_channel(self, curr_channel: slice, level: Optional[int]) -> slice:
    #     return self._set_channel(start = curr_channel.start + 1, stop = curr_channel.stop + 1, step = curr_channel.step, level = level)

    # def _set_channel(self, start: int, stop: Optional[int] = None, step: Optional[int] = None, lvl: Optional[int] = None) -> slice:
    #     if lvl:
    #         max_channel = self._res[lvl]["channels"] - 1
    #     else:
    #         max_channel = self._meta["c_size"] - 1

    #     # start can't surpass max_channel, can't go below 0
    #     start = max(0, min(start, max_channel))

    #     if stop is None:
    #         stop = start + 1
    #     else:
    #         # end must be at least one greater than start
    #         stop = max(start + 1, stop)

    #     if step is None:
    #         step = 1
    #     # step is allowed to exceed max_channel, as long as stop is set properly

    #     return slice(start, stop, step)

    # def _n_slices(self, n: int, lvl: int = 0) -> tuple[slice, slice]:
    #     """0-indexed, assumes row major"""
    #     n_tiles_x = self._res[lvl]["n_tiles_x"]
    #     tile_row = int(n // n_tiles_x)
    #     tile_col = int(n % n_tiles_x)

    #     assert n_tiles_x > 0

    #     row_start = int(tile_row * self._res[lvl]["max_tile_height"])
    #     row_end = int(row_start + self._res[lvl]["max_tile_height"])

    #     col_start = int(tile_col * self._res[lvl]["max_tile_width"])
    #     col_end = int(col_start + self._res[lvl]["max_tile_width"])

    #     assert row_end > row_start
    #     assert col_end > col_start

    #     return (slice(row_start, row_end, 1), slice(col_start, col_end, 1))

    # def _nn(self, lvl: int):
    #     """num of nth values"""
    #     return self._nx(lvl) * self._ny(lvl)

    # def _iy(self, lvl: int, n: int):
    #     """idx of tile in the y direction"""
    #     _tile_width = self._tile_width(lvl)
    #     _img_width = self._res[lvl]["width"]

    #     n_tile_cols = ceil(_img_width / _tile_width)

    #     return n // n_tile_cols

    # def _ny(self, lvl: int):
    #     """num tiles in y direction"""
    #     return self._res[lvl]["n_tiles_y"]

    # def _ix(self, lvl: int, n: int):
    #     """idx of tile in the x direction"""
    #     _tile_width = self._tile_width(lvl)
    #     _img_width = self._res[lvl]["width"]

    #     n_tile_cols = ceil(_img_width / _tile_width)

    #     return n % n_tile_cols

    # def _nx(self, lvl: int):
    #     """num tiles in the x direction"""
    #     return self._res[lvl]["n_tiles_x"]

    # def _tile_height(self, lvl: int):
    #     return self._res[lvl]["max_tile_height"]

    # def _tile_width(self, lvl: int):
    #     return self._res[lvl]["max_tile_width"]

    # @property
    # def _res(self):
    #     meta = self._meta
    #     return meta["resolutions"]

    # @property
    # def _meta(self):
    #     if not self.__cached_meta:
    #         self.__cached_meta = self.__extract_standard_metadata()
    #     return self.__cached_meta

    # @property
    # def _full_meta(self):
    #     if not self.__cached_full_meta:
    #         self.__cached_full_meta = self.__extract_metadata(max_pages=None, include_tags=True)
    #     return self.__cached_full_meta
