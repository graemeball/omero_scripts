#!/usr/bin/env python
"""
impy.py
=========

Image processing helper classes and functions designed to interface
with IPython notebook, OMERO, CellProfiler, CellCognition, Ilastik.
The core data type is the Im class.

"""

__author__ = "Graeme Ball (graemeball@googlemail.com)"
__copyright__ = "Copyright (c) 2013 Graeme Ball"
__license__ = "GPL"  # http://www.gnu.org/licenses/gpl.txt


import numpy as np
import pprint


class Im(object):
    """
    Image object based on a numpy ndarray; OME compatible.

    Attributes
    ----------
        name: image name (string)
        pix : numpy ndarray of pixel data
        dim_order: "CTZYX" (fixed, all images are 5D)
        nc, nt, nz, ny, nx: dimension sizes
        dtype: numpy dtype for pixels
        pixel_size: dict of pixel sizes and units
        ch_info: list of channel info dicts
        description: image description (string)
        tags: dict of tag {'value': description} pairs
        meta_ext: dict of extended metadata {'label': value} pairs

    """

    def __init__(self, pix=None, meta=None):
        """
        Construct Im object using numpy ndarray (pix) and metadata dictionary.
        Default is to construct array of zeros of given size, or 1x1x1x256x256.

        """
        ch_info = [{'label': None, 'em_wave': None, 'ex_wave': None,
                    'color': None}]
        pix_sz = {'x': 1, 'y': 1, 'z': 1, 'units': None}
        default_meta = {'name': "Unnamed", 'dim_order': "CTZYX",
                        'nc': 1, 'nt': 1, 'nz': 1, 'ny': 256, 'nx': 256,
                        'dtype': np.uint8, 'pixel_size': pix_sz,
                        'ch_info': ch_info, 'description': "",
                        'tags': {}, 'meta_ext': {}}
        for key in default_meta:
            setattr(self, key, default_meta[key])
        if meta is not None:
            for key in meta:
                setattr(self, key, default_meta[key])
        if pix is None:
            # default, construct empty 8-bit image according to dimensions
            self.pix = np.zeros((self.nc, self.nt, self.nz, self.ny, self.nx),
                                dtype=np.uint8)
        else:
            if isinstance(pix, np.ndarray):
                self.pix = pix
            else:
                raise TypeError("pix must be " + str(np.ndarray))
            self.nc, self.nt, self.nz, self.ny, self.nx = pix.shape
            self.dtype = pix.dtype

    def __repr__(self):
        im_repr = 'Im object "{0}"\n'.format(self.name)
        im_repr += pprint.pformat(vars(self))
        return im_repr
