#!/usr/bin/env python
"""
iomero.py
=========

A module for interfacing with OMERO (I/O), including both interactive
work and command line usage. Configure by editing SETUP CONSTANTS.

For interactive use from a python shell, create an Omg instance.

"""

__author__ = "Graeme Ball (graemeball@googlemail.com)"
__copyright__ = "Copyright (c) 2013 Graeme Ball"
__license__ = "GPL"  # http://www.gnu.org/licenses/gpl.txt

# SETUP CONSTANTS
OMERO_PYTHON = "/Users/graemeb/build/OMERO.server/lib/python"
ICE_PATH = "/usr/local/Cellar/zeroc-ice34/3.4.2/python"
SERVER = "localhost"  # default
PORT = 4064           # default


import os
import sys
import argparse
sys.path.append(OMERO_PYTHON)
sys.path.append(ICE_PATH)
from omero.gateway import BlitzGateway
import omero.cli


class Omg(object):
    """
    OMERO gateway that wraps Blitz gateway and CLI, intended for
    scripting and interactive work.

    Attributes
    ----------
    conn : Blitz gateway connection

    """

    def __init__(self, conn=None, user=None, passwd=None,
                 server=SERVER, port=PORT, skey=None):
        """
        Requires active Blitz connection OR username plus password or sesskey
        """
        if conn is None and (user is None or (passwd is None and skey is None)):
            raise ValueError("Bad parameters," + self.__init__.__doc__)
        if conn is not None:
            if conn.isConnected():
                self.conn = conn
            else:
                raise ValueError("Cannot initialize Omg with a closed connection!")
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

    def put(self, filename, name=None, dataset=None):
        """
        Import filename usign OMERO CLI, optionally with a specified name
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

    def ls(self):
        """
        Print a listing of all groups, projects, datasets, images.
        """
        def ls_groups():
            groups = self.conn.getGroupsMemberOf()
            return [(group.getId(), group.getName()) for group in groups]

        def ls_projects(group_id):
            projs = self.conn.listProjects(self.conn.getUserId())
            return [(proj.getId(), proj.getName()) for proj in projs]

        def ls_datasets(proj_id):
            dsets = self.conn.getObject("Project", proj_id).listChildren()
            return [(dset.getId(), dset.getName()) for dset in dsets]

        def ls_images(dset_id):
            imgs = self.conn.getObject("Dataset", dset_id).listChildren()
            return [(img.getId(), img.getName()) for img in imgs]

        for gid, gname in ls_groups():
            print("Group: {0} ({1})".format(gname, str(gid)))
            for pid, pname in ls_projects(gid):
                print("  Project: {0} ({1})".format(pname, str(pid)))
                for did, dname in ls_datasets(pid):
                    print("    Dataset: {0} ({1})".format(dname, str(did)))
                    for iid, iname in ls_images(did):
                        print("      Image: {0} ({1})".format(iname, str(iid)))

    def get(self, im_id):
        """
        Download the specified image as an OME-TIFF to current directory.
        Return : path to downloaded image
        """
        img = self.conn.getObject("Image", oid=im_id)
        path_and_base, ext = os.path.splitext(img.getName())
        base = os.path.basename(path_and_base)  # name in OMERO can has path
        img_name = "{0}_{1}.ome.tiff".format(base, str(im_id))
        img_path = os.path.join(os.getcwd(), img_name)
        img_file = open(str(img_path), "wb")
        fsize, blockgen = img.exportOmeTiff(bufsize=65536)
        for block in blockgen:
            img_file.write(block)
        img_file.close()
        return img_path


class _FriendlyParser(argparse.ArgumentParser):
    """
    Display help message upon incorrect args -- no unfriendly error messages.
    """
    def error(self, message):
        print(self.print_help())
        print(message)
        sys.exit(2)


def _main():
    """
    'iomero': a command line OMERO importer with Blitz gateway extras
    """
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