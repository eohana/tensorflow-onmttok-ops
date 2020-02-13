from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

onmttok_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_tensorflow_onmttok_ops.so'))

detokenize = onmttok_ops.detokenize
tokenize = onmttok_ops.tokenize
