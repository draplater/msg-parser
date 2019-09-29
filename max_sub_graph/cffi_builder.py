import random
import string

import os

import cffi
import sys

import shutil


rnd = random.Random()


if not os.path.exists("libs"):
    try:
        os.mkdir("libs")
    except OSError:
        pass


def build_1ec2p():
    ffi = cffi.FFI()
    ffi.cdef('int parse(int num_point, double Score[][256][2], int arcs[][2]);')

    current_path = os.path.dirname(__file__)
    with open(os.path.join(current_path, "src/1ec2p.cpp")) as f:
        ffi.set_source("lib1ec2p", f.read(), source_extension='.cpp', extra_compile_args=['-std=c++11', "-O3"])
    dir_name = "/tmp/libs_" + "".join(rnd.choice(string.letters + string.digits) for i in range(0, 10))
    ffi.compile(verbose=False, tmpdir=dir_name)

    sys.path.insert(0, dir_name)
    # noinspection PyUnresolvedReferences
    from lib1ec2p import lib as lib1ec2p
    shutil.rmtree(dir_name, ignore_errors=True)
    sys.path.pop(0)

    return lib1ec2p, ffi


def build_1ec2p_vine():
    ffi = cffi.FFI()
    ffi.cdef('int parse_vine(int max_arc_len, int num_point, double Score[][256][2], int arcs[][2]);')

    current_path = os.path.dirname(__file__)
    with open(os.path.join(current_path, "src/1ec2p_vine.cpp")) as f:
        ffi.set_source("lib1ec2p_vine", f.read(), source_extension='.cpp', extra_compile_args=['-std=c++11', "-O3"])
    dir_name = "/tmp/libs_" + "".join(rnd.choice(string.letters + string.digits) for i in range(0, 10))
    ffi.compile(verbose=False, tmpdir=dir_name)

    sys.path.insert(0, dir_name)
    # noinspection PyUnresolvedReferences
    from lib1ec2p_vine import lib as lib1ec2p
    sys.path.pop(0)
    shutil.rmtree(dir_name, ignore_errors=True)

    return lib1ec2p, ffi
