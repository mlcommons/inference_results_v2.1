from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MF-MLCommons")

# Base paths
BASE_CODE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = Path(__file__).resolve().parent.parent.parent

def wrap_function(lib, funcname, argtypes, restype):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func
