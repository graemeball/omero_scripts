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
        ch_info: list of channel info dicts
        pixel_size: dict of pixel sizes and units
        description: image description (string)
        tags: dict of tag {'value': description} pairs
        meta_ext: dict of extended metadata {'label': value} pairs

    """

    def __init__(self, pix=None, meta=None):
        """
        Construct Im object using numpy ndarray (pix) and metadata dictionary.
        Default is to construct array of zeros of given size, or 1x1x1x256x256.

        """
        if meta is None:
            ch_info = [{'label': None, 'em_wave': None, 'ex_wave': None,
                        'color': None}]
            meta = {'name': "Unnamed", 'dim_order': "CTZYX",
                    'nc': 1, 'nt': 1, 'nz': 1, 'ny': 256, 'nx': 256,
                    'dtype': np.uint8, 'ch_info': ch_info, 'pixel_size': 1,
                    'description': "", 'tags': {}, 'meta_ext': {}}
        for val, key in enumerate(meta):
            setattr(self, key, val)
        if pix is None:
            # default, construct empty 8-bit image according to dimensions
            self.pix = np.zeros((self.nc, self.nt, self.nz, self.ny, self.nx),
                                dtype=np.uint8)
        else:
            self.pix = pix
            self.nc, self.nt, self.nz, self.ny, self.nx = pix.shape

    def __repr__(self):
        im_repr = 'Im object "{0}"\n'.format(self.name)
        im_repr += pprint.pformat(vars(self))
        return im_repr
