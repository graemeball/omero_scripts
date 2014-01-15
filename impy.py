#!/usr/bin/env python
"""
impy.py
=========

Im image class to simplify scripting with OMERO and IPython notebook.

"""
# TODO: 
#   - add helper methods 
#   - add interfaces to other python image processing tools
#   - more metadata
#   - large files / memmap / hdf5
#   - parallel / locking

__author__ = "Graeme Ball (graemeball@googlemail.com)"
__copyright__ = "Copyright (c) 2013 Graeme Ball"
__license__ = "GPL"  # http://www.gnu.org/licenses/gpl.txt


import numpy as np
import pprint


class Im(object):
    """
    Image object based on numpy ndarray, with easily accessible core metadata.

    Attributes
    ----------
        name: image name (string)
        pix : numpy ndarray of pixel data
        dim_order: "CTZYX" (fixed, all images are 5D)
        dtype: numpy dtype for pixels
        shape: pixel array shape tuple, (nc, nt, nz, ny, nx)
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
        channels = [{'label': None, 'em_wave': None, 'ex_wave': None, 'color': None}]
        pixel_size = {'x': 1, 'y': 1, 'z': 1, 'units': None}
        default_meta = {'name': "Unnamed", 'dim_order': "CTZYX",
                        'dtype': np.uint8, 'shape': (1, 1, 1, 256, 256),
                        'pixel_size': pixel_size, 'channels': channels,
                        'description': "", 'tags': {}, 'meta_ext': {}}
        for key in default_meta:
            setattr(self, key, default_meta[key])
        if meta is not None:
            for key in meta:
                setattr(self, key, meta[key])
        if pix is None:
            # default, construct empty 8-bit image according to dimensions
            self.pix = np.zeros((self.nc, self.nt, self.nz, self.ny, self.nx),
                                dtype=np.uint8)
        else:
            if isinstance(pix, np.ndarray):
                self.pix = pix
            else:
                raise TypeError("pix must be " + str(np.ndarray))
            self.shape = pix.shape
            self.dtype = pix.dtype

    def __repr__(self):
        im_repr = 'Im object "{0}"\n'.format(self.name)
        im_repr += pprint.pformat(vars(self))
        return im_repr
