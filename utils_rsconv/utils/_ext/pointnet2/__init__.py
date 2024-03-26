
from torch.utils.ffi import _wrap_function
from ._pointnet2 import lib as _lib, ffi as _ffi

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())

# import torch
# from torch.utils.cpp_extension import load

# # Assuming '_pointnet2.cpp' is the C++ source file containing your functions

# # Define the extension module
# _pointnet2 = load(
#     'pointnet2', ['_pointnet2.cpp'],
#     verbose=True, with_cuda=False
# )

# # Import symbols from the extension module
# __all__ = []
# def _import_symbols(locals):
#     for symbol in dir(_pointnet2):
#         fn = getattr(_pointnet2, symbol)
#         locals[symbol] = fn
#         __all__.append(symbol)

# _import_symbols(locals())
