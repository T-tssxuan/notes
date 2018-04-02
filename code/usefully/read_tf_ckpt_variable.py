from tensorflow.python import pywrap_tensorflow

from tensorflow.python.tools import inspect_checkpoint as chkp

def get_all_variable(fname, vname):
  reader = pywrap_tensorflow.NewCheckpointReader(fname)
  return reader.get_tensor(key)

def get_all_variable_name(fname):
  reader = pywrap_tensorflow.NewCheckpointReader(fname)
  var_to_shape_map = reader.get_variable_to_shape_map()
  return sorted(var_to_shape_map)

chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)
