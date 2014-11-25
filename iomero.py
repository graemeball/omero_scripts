#!/usr/bin/env python
"""
iomero.py
=========

A module for interfacing with OMERO (I/O), including both interactive
work and command line usage. Configure by editing SETUP CONSTANTS.

Create an Omg OMERO gateway instance to interact with OMERO.
The Im class provides easy access to image pixel data and core metadata.

"""

__author__ = "Graeme Ball (graemeball@googlemail.com)"
__copyright__ = "Copyright (c) 2013 Graeme Ball"
__license__ = "GPL"  # http://www.gnu.org/licenses/gpl.txt

# SETUP CONSTANTS
OMERO_PYTHON = "/usr/local/Cellar/omero/5.0.0-rc1/lib/python"
#ICE_PATH = "no longer needed on Mac as of Ice3.5"
SERVER = "localhost"  # default
PORT = 4064           # default


import os
import sys
import argparse
import pprint
import numpy as np
import getpass
sys.path.append(OMERO_PYTHON)
#sys.path.append(ICE_PATH)
import omero.cli
import omero.model
import omero.rtypes
from omero.gateway import BlitzGateway
from omero.util import script_utils
import pylab, matplotlib


class Omg(object):
    """
    OMERO gateway that wraps Blitz gateway and CLI, intended for
    scripting and interactive work.

    Attributes
    ----------
    conn : Blitz gateway connection

    """

    def __init__(self, conn=None, user=None, server=SERVER, port=PORT, skey=None):
        """
        Requires active Blitz connection OR username plus password or sesskey
        """
        passwd = None
        if skey is None:
            passwd = getpass.getpass()
        # TODO: clean this up!!
        if conn is None and (user is None or (passwd is None and skey is None)):
            raise ValueError("Bad parameters," + self.__init__.__doc__)
        if conn is not None:
            if conn.isConnected():
                self.conn = conn
            else:
                raise ValueError("Cannot initialize with closed connection!")
        else:
            if passwd is not None:
                self.conn = BlitzGateway(user, passwd, host=server, port=port)
                self.conn.connect()
            else:
                self.conn = BlitzGateway(user, host=server, port=port)
                self.conn.connect(skey)
        if self.conn.isConnected():
            self._server = self.conn.host
            self._port = self.conn.port
            self._user = self.conn.getUser().getName()
            self._key = self.conn.getSession().getUuid().getValue()
            print("Connected to {0} (port {1}) as {2}, session key={3}".format(
                  self._server, self._port, self._user, self._key))
        else:
            print("Failed to open connection :-(")

    def ls(self, uid=None):
        """
        Print groups, then projects/datasets/images for current group.
        """
        print("Groups for {0}:-".format(self.conn.getUser().getName()))
        for gid, gname in self._ls_groups():
            print("  {0} ({1})".format(gname, str(gid)))
        curr_grp = self.conn.getGroupFromContext()
        gid, gname = curr_grp.getId(), curr_grp.getName()
        print("\nData for current group, {0} ({1}):-".format(gname, gid))
        if uid is None:
            uid = self.conn.getUserId()
        for pid, pname in self._ls_projects(uid=uid):
            print("  Project: {0} ({1})".format(pname, str(pid)))
            for did, dname in self._ls_datasets(pid):
                print("    Dataset: {0} ({1})".format(dname, str(did)))
                for iid, iname in self._ls_images(did):
                    print("      Image: {0} ({1})".format(iname, str(iid)))
        # TODO, list orphaned Datasets and Images

    def _ls_groups(self):
        """list groups (id, name) this session is a member of"""
        groups = self.conn.getGroupsMemberOf()
        return [(group.getId(), group.getName()) for group in groups]

    def _ls_projects(self, uid=None):
        """list projects (id, name) in the current session group"""
        if uid is not None:
            projs = self.conn.listProjects(uid)
        else:
            projs = self.conn.listProjects()
        return [(proj.getId(), proj.getName()) for proj in projs]

    def _ls_datasets(self, proj_id):
        """list datasets (id, name) within the project id given"""
        dsets = self.conn.getObject("Project", proj_id).listChildren()
        return [(dset.getId(), dset.getName()) for dset in dsets]

    def _ls_images(self, dset_id):
        """list images (id, name) within the dataset id given"""
        imgs = self.conn.getObject("Dataset", dset_id).listChildren()
        return [(img.getId(), img.getName()) for img in imgs]

    def groups(self):
        """
        List all groups with group Ids.
        """
        groups = [group.getName() + " (" + str(group.getId()) + ")"
                  for group in self.conn.listGroups()]
        return sorted(groups)

    def users(self):
        """
        List all users with user Ids.
        """
        def isAd(user):
            user = self.conn.getObject("Experimenter", oid=user.getId())
            if user.isAdmin():
                return "A"
            else:
                return ""

        users = [user.getName() + " (" + str(user.getId()) + isAd(user) + ")"
                 for user in self.conn.findExperimenters()]
        return sorted(users)

    def rois(self, iid):
        """
        Return list of ROIs for a given image id (iid).
        """
        rois = []
        result = self.conn.getRoiService().findByImage(iid, None)
        for roi in result.rois:
            rois.append(roi)
        return rois

    def copy_rois(self, iid_src, iid_dest):
        """
        Copy all ROIs from image iid_src to iid_dest.
        """
        img_src = self.conn.getObject("Image", iid_src)
        img_dest = self.conn.getObject("Image", iid_dest)
        # TODO: warn if dimensions do not match
        us = self.conn.getUpdateService()

        def _clone_shape(shape):
            # avert your eyes...
            shape_type = type(shape)
            new_shape = shape_type()
            getters = [m for m in dir(shape) if m[0:3] == "get"]
            getters.remove("getId")
            getters.remove("getDetails")
            for meth in getters:
                stuff = eval("shape." + meth + "()")
                eval("new_shape.set" + meth[3:] + "(stuff)")
            return new_shape

        for roi in self.rois(iid_src):
            roi_new = omero.model.RoiI()
            roi_new.setImage(img_dest._obj)
            for shape in roi.copyShapes():
                roi_new.addShape(_clone_shape(shape))
            r = us.saveAndReturnObject(roi_new)

    def ustats(self):
        """
        Report per user statistics: projects, datasets, images, bytes.
        """
        users = [user for user in self.conn.findExperimenters()]
        stats = []
        tproj, tdset, tim, tbyte = 0, 0, 0, 0
        for user in users:
            nproj, ndset, nim, nbyte = 0, 0, 0, 0
            for pid, pname in self._ls_projects(user.getId()):
                nproj += 1
                for did, dname in self._ls_datasets(pid):
                    ndset += 1
                    for iid, iname in self._ls_images(did):
                        nim += 1
                        nbyte += self._nbytes(iid)
            stat = "{0} ({1}): ".format(user.getName(), user.getId())
            stat += "{0} projects, {1} datasets, {2} images, {3} bytes".format(
                    nproj, ndset, nim, nbyte)
            stats.append(stat)
            tproj += nproj
            tdset += ndset
            tim += nim
            tbyte += nbyte
        stat = "~TOTAL: "
        stat += "{0} projects, {1} datasets, {2} images, {3} GB".format(
                tproj, tdset, tim, tbyte * 1.0 / 1E9)
        stats.append(stat)
        return sorted(stats)

    def _nbytes(self, im_id):
        """Estimate number of bytes in an image"""
        iobj = self.conn.getObject("Image", oid=im_id)
        npix = iobj.getSizeC() * iobj.getSizeT() * iobj.getSizeZ()
        npix *= iobj.getSizeY() * iobj.getSizeX()
        bytes_per_pix = np.dtype(iobj.getPixelsType()).itemsize
        return npix * bytes_per_pix

    def chgrp(self, group_id):
        """
        Change group for this session to the group_id given.
        """
        self.conn.setGroupForSession(group_id)

    def get(self, im_id, get_att=True):
        """
        Download the specified image as an OME-TIFF to current directory,
        with attachments also downloaded to folder: img_path + '_attachments'
        Return : path to downloaded image
        """
        img = self.conn.getObject("Image", oid=im_id)
        img_name = self._unique_name(img.getName(), im_id)
        img_path = os.path.join(os.getcwd(), img_name) + ".ome.tiff"
        img_file = open(img_path, "wb")
        fsize, blockgen = img.exportOmeTiff(bufsize=65536)
        for block in blockgen:
            img_file.write(block)
        img_file.close()
        fa_type = omero.model.FileAnnotationI
        attachments = [ann for ann in img.listAnnotations()
                       if ann.OMERO_TYPE == fa_type]
        if get_att and len(attachments) > 0:
            att_dir = img_path + "_attachments"
            os.mkdir(att_dir)

            def download_attachment(att, att_dir):
                """download OMERO file annotation to att_dir"""
                att_file = open(os.path.join(att_dir, att.getFileName()), "wb")
                for att_chunk in att.getFileInChunks():
                    att_file.write(att_chunk)
                att_file.close()

            for att in attachments:
                download_attachment(att, att_dir)
        return img_path

    def cp(self, im_id):
        """
        Create a copy of an image in the same dataset, named name_id_CPY.
        Set description to a note of original group,proj,dset,image
        and return id of the new image copy.
        """
        img = self.conn.getObject("Image", im_id)

        # re-using PrimaryPixels means img_cpy is stuck in original group :-(
        #zct = [(z, c, t) for t in range(img.getSizeT())
        #                 for c in range(img.getSizeC())
        #                 for z in range(img.getSizeZ())]
        #pplanes = img.getPrimaryPixels().getPlanes(zct)
        #name_cpy = self._unique_name(img.getName(), im_id) + "_CPY"
        #img_cpy = self.conn.createImageFromNumpySeq(
        #        pplanes, name_cpy, sourceImageId=im_id)

        # this is suboptimal...
        im_path = self.get(im_id)
        cpy_name = self._unique_name(img.getName(), im_id) + "_CPY"
        img_cpy = self.put(im_path, name=cpy_name,
                           dataset=img.getParent().getId())

        # identify original source image in description
        origin = "\nImage: {0} ({1})".format(img.getName(), img.getId())
        origin += "\nDataset: " + str(img.getParent().getName())
        origin += "\nProject: " + str(img.getParent().getParent().getName())
        origin += "\nGroup: " + self.conn.getGroupFromContext().getName()
        self.describe(img_cpy, "Copied from..." + origin)
        return img_cpy

    def _unique_name(self, img_name, im_id):
        """Make unique name combining a file basename & OMERO Image id"""
        path_and_base, ext = os.path.splitext(img_name)
        base = os.path.basename(path_and_base)  # name in OMERO can has path
        return "{0}_{1}".format(base, str(im_id))

    def dget(self, dataset_id):
        """
        Download an entire OMERO Dataset to the current directory.
        """
        downloads = []
        wdir = os.getcwd()
        dset_name = self.conn.getObject("Dataset", dataset_id).getName()
        dset_path = os.path.join(wdir, dset_name + "_D" + str(dataset_id))
        os.mkdir(dset_path)
        os.chdir(dset_path)
        for img_id, img_name in self._ls_images(dataset_id):
            downloads.append(self.get(img_id))
        os.chdir(wdir)
        return downloads

    def pget(self, project_id):
        """
        Download an entire OMERO Project to the current directory.
        """
        downloads = []
        wdir = os.getcwd()
        proj_name = self.conn.getObject("Project", project_id).getName()
        proj_path = os.path.join(wdir, proj_name + "_P" + str(project_id))
        os.mkdir(proj_path)
        os.chdir(proj_path)
        for dset_id, dset_name in self._ls_datasets(project_id):
            downloads.extend(self.dget(dset_id))
        os.chdir(wdir)
        return downloads

    def put(self, filename, name=None, dataset=None):
        """
        Import filename using OMERO CLI, optionally with a specified name
        to a specified dataset (dataset_id).
        Return : OMERO image Id
        """
        cli = omero.cli.CLI()
        cli.loadplugins()
        import_args = ["import"]
        import_args.extend(["-s", str(self._server)])
        import_args.extend(["-k", str(self._key)])
        if dataset is not None:
            import_args.extend(["-d", str(dataset)])
        if name is not None:
            import_args.extend(["-n", str(name)])
        clio = "cli.out"
        clie = "cli.err"
        import_args.extend(["---errs=" + clie, "---file=" + clio, "--"])
        import_args.append(filename)
        cli.invoke(import_args, strict=True)
        pix_id = int(open(clio, 'r').read().rstrip())
        im_id = self.conn.getQueryService().get("Pixels", pix_id).image.id.val
        os.remove(clio)
        os.remove(clie)
        return im_id

    def describe(self, im_id, description):
        """
        Append to image description.
        """
        img = self.conn.getObject("Image", oid=im_id)
        old_description = img.getDescription() or ""
        img.setDescription(old_description + "\n" + description)
        img.save()

    def attach(self, im_id, attachments):
        """
        Attach a list of files to an image.
        """
        img = self.conn.getObject("Image", oid=im_id)
        for attachment in attachments.split():
            fann = self.conn.createFileAnnfromLocalFile(attachment)
            img.linkAnnotation(fann)
        img.save()

    # TODO: ls_tags() and tag() methods?

    def mkp(self, project_name, description=None):
        """
        Make new OMERO project in current group, returning the new project Id.
        """
        # see: omero/lib/python/omeroweb/webclient/controller/container.py
        proj = omero.model.ProjectI()
        proj.name = omero.rtypes.rstring(str(project_name))
        if description is not None and description != "":
            proj.description = omero.rtypes.rstring(str(description))
        return self._save_and_return_id(proj)

    def mkd(self, dataset_name, project_id=None, description=None):
        """
        Make new OMERO dataset, returning the new dataset Id.
        """
        dset = omero.model.DatasetI()
        dset.name = omero.rtypes.rstring(str(dataset_name))
        if description is not None and description != "":
            dset.description = omero.rtypes.rstring(str(description))
        if project_id is not None:
            l_proj_dset = omero.model.ProjectDatasetLinkI()
            proj = self.conn.getObject("Project", project_id)
            l_proj_dset.setParent(proj._obj)
            l_proj_dset.setChild(dset)
            dset.addProjectDatasetLink(l_proj_dset)
        return self._save_and_return_id(dset)

    def _save_and_return_id(self, obj):
        """Save new omero object and return id assgined to it"""
        # see: OmeroWebGateway.saveAndReturnId
        # in: lib/python/omeroweb/webclient/webclient_gateway.py
        u_s = self.conn.getUpdateService()
        res = u_s.saveAndReturnObject(obj, self.conn.SERVICE_OPTS)
        res.unload()
        return res.id.val

    def im(self, im_id):
        """
        Return an Im object for the image id specified.
        """
        img = self.conn.getObject("Image", im_id)
        # build pixel np.ndarray
        nx, ny = img.getSizeX(), img.getSizeY()
        nz, nt, nc = img.getSizeZ(), img.getSizeT(), img.getSizeC()
        planes = [(z, c, t) for c in range(nc)
                  for t in range(nt)
                  for z in range(nz)]
        pix_gen = img.getPrimaryPixels().getPlanes(planes)
        pix = np.array([i for i in pix_gen]).reshape((nc, nt, nz, ny, nx))
        # initialize Im using pix and extracted metadata
        meta = self._extract_meta(img, im_id)
        return Im(pix=pix, meta=meta)

    def _extract_meta(self, img, im_id):
        """Extract metadata attributes from OMERO Blitz gateway Image"""
        meta = {}
        meta['name'] = self._unique_name(img.getName(), im_id)
        meta['description'] = img.getDescription()

        def _extract_ch_info(ch):
            """extract core metadata for for channel, return as dict"""
            ch_info = {'label': ch.getLabel()}
            ch_info['ex_wave'] = ch.getExcitationWave()
            ch_info['em_wave'] = ch.getEmissionWave()
            ch_info['color'] = ch.getColor().getRGB()
            return ch_info

        meta['channels'] = [_extract_ch_info(ch) for ch in img.getChannels()]
        meta['pixel_size'] = {'x': img.getPixelSizeX(),
                              'y': img.getPixelSizeY(),
                              'z': img.getPixelSizeZ(),
                              'units': "um"}
        tag_type = omero.model.TagAnnotationI
        tags = [ann for ann in img.listAnnotations()
                if ann.OMERO_TYPE == tag_type]
        meta['tags'] = {tag.getValue() + " (" + str(tag.getId()) + ")":
                        tag.getDescription() for tag in tags}
        fa_type = omero.model.FileAnnotationI
        attachments = [ann for ann in img.listAnnotations()
                       if ann.OMERO_TYPE == fa_type]
        meta['attachments'] = [att.getFileName() + " (" + str(att.getId()) +
                               ")" for att in attachments]
        user_id = self.conn.getUser().getName() + " (" + \
            str(self.conn.getUser().getId()) + ") @" + self.conn.host
        meta_ext = {}
        meta_ext['user_id'] = user_id
        meta['meta_ext'] = meta_ext
        # TODO: ROIs, display settings?
        # objective: Image.loadOriginalMetadata()[1][find 'Lens ID Number'][1],
        return meta

    def imput(self, im, dataset_id=None):
        """
        Create a new OMERO Image using an Im object, returning new image id.
        """
        # see: omero/lib/python/omero/util/script_utils.py
        # see: omero/lib/python/omeroweb/webclient/webclient_gateway.py
        # see: https://gist.github.com/will-moore/4141708
        if not isinstance(im, Im):
            raise TypeError("first imput argument must be of type Im")
        nc, nt, nz, ny, nx = im.shape
        ch_nums = range(nc)
        q_s = self.conn.getQueryService()
        p_s = self.conn.getPixelsService()
        c_s = self.conn.getContainerService()
        u_s = self.conn.getUpdateService()
        pu_s = self.conn.c.sf.createRawPixelsStore()
        q_ptype = "from PixelsType as p where p.value='{0}'".format(
                  str(im.dtype))
        pixelsType = q_s.findByQuery(q_ptype, None)
        im_id = p_s.createImage(nx, ny, nz, nt, ch_nums, pixelsType,
                    im.name, im.description)
        img_i = c_s.getImages("Image", [im_id.getValue()], None)[0]
        img = self.conn.getObject("Image", im_id.getValue())
        pix_id = img_i.getPrimaryPixels().getId().getValue()
        pu_s.setPixelsId(pix_id, True)
        for c in range(nc):
            for t in range(nt):
                for z in range(nz):
                    plane = im.pix[c, t, z, :, :]
                    # TODO, convert to createImageFromNumpySeq
                    script_utils.uploadPlaneByRow(pu_s, plane, z, c, t)
        l_dset_im = omero.model.DatasetImageLinkI()
        dset = self.conn.getObject("Dataset", dataset_id)
        l_dset_im.setParent(dset._obj)
        l_dset_im.setChild(img._obj)
        self._update_meta(im, im_id)
        u_s.saveObject(l_dset_im, self.conn.SERVICE_OPTS)
        return im_id.getValue()

    def _update_meta(self, im, im_id):
        """Set OMERO Image metadata using Im metadata"""
        # TODO: store im metadata in OMERO image with im_id
        # channels [{'label': label, 'ex_wave': ex_wave,
        #           'em_wave': em_wave, 'color': RGB},...]
        # pixel_size [{'x': psx, 'y': psy, 'z': psz, 'units': ?},...]
        # tags {'tag1 (id1)': 'desc1',...}
        # user_id
        # objective, attachments, ROIs, display settings


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
        attachments: list of attachment files
        meta_ext: dict of extended metadata {'label': value} pairs

    """

    def __init__(self, pix=None, meta=None):
        """
        Construct Im object using numpy ndarray (pix) and metadata dictionary.
        Default is to construct array of zeros of given size, or 1x1x1x256x256.

        """
        channels = [{'label': None, 'em_wave': None, 'ex_wave': None,
                     'color': None}]
        pixel_size = {'x': 1, 'y': 1, 'z': 1, 'units': None}
        default_meta = {'name': "Unnamed", 'dim_order': "CTZYX",
                        'dtype': np.uint8, 'shape': (1, 1, 1, 256, 256),
                        'pixel_size': pixel_size, 'channels': channels,
                        'description': "", 'tags': {}, 'attachments': [],
                        'meta_ext': {}}
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


# custom ArgumentParser
class _FriendlyParser(argparse.ArgumentParser):
    """
    Display help message upon incorrect args -- no unfriendly error messages.
    """
    def error(self, message):
        """override parser error handling"""
        print(self.print_help())
        print(message)
        sys.exit(2)


def _main():
    """'iomero': a command line OMERO importer with Blitz gateway extras"""
    parser = _FriendlyParser(description=_main.__doc__)
    parser.add_argument('server', help='OMERO server name / IP address')
    parser.add_argument('-u', '--user', action="store", type=str,
        required=True, help="OMERO experimenter name (username)")
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument('-w', '--password', action="store", type=str,
        help="OMERO experimenter password")
    credentials.add_argument('-k', '--sesskey', action="store", type=str,
        help="OMERO session key (replaces -w)")
    parser.add_argument('-f', '--filename', action="store", type=str,
        required=True, help="Image file to upload")
    parser.add_argument('-d', '--dataset', action="store", type=str,
        required=False, help="OMERO dataset Id to import image into")
    parser.add_argument('-n', '--name', action="store", type=str,
        required=False, help="Image name to use")
    parser.add_argument('-i', '--info', action="store", type=str,
        required=False, help="Image description to use")
    parser.add_argument('-a', '--attach', action="store", type=str,
        required=False, help='Attach files in quotes (e.g. "file1 file2")')
    parser.add_argument('-p', '--port', action="store", type=int,
        required=False, help="OMERO server port, defaults to ")
    args = parser.parse_args()

    # create an Omg OMERO gateway
    init_args = {'server': args.server, 'user': args.user}
    if args.password:
        init_args['passwd'] = args.password
    else:
        init_args['skey'] = args.sesskey
    if args.port:
        init_args['port'] = args.port
    omg = Omg(**init_args)

    # handle file upload (file, dataset, name)
    put_args = {'filename': args.filename}
    if args.name:
        put_args['name'] = args.name
    if args.dataset:
        put_args['dataset'] = args.dataset
    im_id = omg.put(**put_args)

    # handle annotation and attachments
    if args.info:
        omg.describe(im_id, args.info)
    if args.attach:
        omg.attach(im_id, args.attach)


if __name__ == "__main__":
    _main()
