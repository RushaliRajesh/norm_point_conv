import glob
import torch
import os.path as osp
# from torch.utils.ffi import create_extension
import sys, argparse, shutil
import glob
import os
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

base_dir = osp.dirname(osp.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for building pointnet2 ffi extension"
    )
    parser.add_argument("--objs", nargs="*")
    clean_arg = parser.add_mutually_exclusive_group()
    clean_arg.add_argument("--build", dest='build', action="store_true")
    clean_arg.add_argument("--clean", dest='clean', action="store_true")
    parser.set_defaults(build=False, clean=False)

    args = parser.parse_args()
    assert args.build or args.clean

    return args

def build(args):
    extra_objects = args.objs
    extra_objects += [a for a in glob.glob('/usr/local/cuda/lib64/*.a')]

    setup(
        name='_ext.pointnet2',
        ext_modules=[
            CUDAExtension(
                name='_ext.pointnet2',
                sources=[a for a in glob.glob("csrc/*.cpp")],
                define_macros=[('WITH_CUDA', None)],
                include_dirs=[osp.join(os.getcwd(), 'cinclude')],
                extra_objects=extra_objects
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        script_args=['build_ext']
    )



def clean(args):
    shutil.rmtree(osp.join(base_dir, "_ext"))


if __name__ == "__main__":
    args = parse_args()
    if args.clean:
        clean(args)
    else:
        build(args)