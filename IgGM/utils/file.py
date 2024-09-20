# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import io
import json
import os
import tempfile
import urllib.parse as urlparse

import requests


def get_tmp_dpath():
    """Get the directory path to temporary files.

    Args: n/a

    Returns:
        tmp_dpath: directory path to temporary files
    """
    if 'TMPDIR_PRFX' not in os.environ:
        print('environmental variable <TMPDIR_PRFX> not defined')
        print('using default <TMPDIR_PRFX>')
        tmp_dpath_prfx = os.getcwd() + '/tmp'
    else:
        tmp_dpath_prfx = os.getenv('TMPDIR_PRFX')

    os.makedirs(tmp_dpath_prfx, exist_ok=True)
    tmp_dpath = tempfile.TemporaryDirectory(dir=tmp_dpath_prfx).name  # pylint: disable=consider-using-with
    os.makedirs(tmp_dpath, exist_ok=True)

    return tmp_dpath


def filename_from_url(url):
    """:return: detected filename as unicode or None"""
    # [ ] test urlparse behavior with unicode url
    fname = os.path.basename(urlparse.urlparse(url).path)
    if len(fname.strip(" \n\t.")) == 0:
        return None

    return fname


def download_file(url, output=None, overwrite=False, max_try_times=10):
    if output is None:
        output = filename_from_url(url) or 'download.IgGM'

    if os.path.isfile(output) and not overwrite:
        print(f"exist file: {output}")
        return output

    ok = False
    for i in range(max_try_times):
        response = requests.get(url)
        if response.status_code != 200:
            continue

        with open(output, mode="wb") as f:
            f.write(response.content)
            ok = True
            break

    if not ok:
        print(f"failed to download {url}")
    return output if ok else None


def jload(f, mode='r', object_pairs_hook=None):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f, object_pairs_hook=object_pairs_hook)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A file handle or string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    is_fio = isinstance(f, io.IOBase)
    if not is_fio:
        folder = os.path.dirname(f)
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        f = open(f, mode=mode)

    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)

    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")

    if not is_fio:
        f.close()
