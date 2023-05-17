import tensorflow as tf
import cupy
import os

def setup_cuda(
    device_num : str,
    verbose : bool = False
    ):
		
	os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
		
	gpus =  tf.config.list_logical_devices('GPU')
	strategy = tf.distribute.MirroredStrategy(gpus)

	physical_devices = tf.config.list_physical_devices('GPU')
	
	for device in physical_devices:	

		try:
			tf.config.experimental.set_memory_growth(device, True)
		except:
			# Invalid device or cannot modify virtual devices once initialized.
			pass
	
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	if verbose:
		print(tf.config.list_physical_devices("GPU"))
		
	return strategy

def cupy_to_tensor(cupy_array):
    # Convert CuPy array to DLPack
    dlpack = cupy_array.toDlpack()

    # Convert DLPack to TensorFlow tensor
    tensor = tf.experimental.dlpack.from_dlpack(dlpack)

    return tensor

def tensor_to_cupy(tensor):
    """
    This function converts a TensorFlow tensor to a CuPy array.

    Parameters:
    tensor (tensorflow.Tensor): a TensorFlow tensor

    Returns:
    cupy.ndarray: a CuPy array
    """
    # Convert TensorFlow tensor to DLPack capsule
    dlpack = tf.experimental.dlpack.to_dlpack(tensor)

    # Convert DLPack capsule to CuPy array
    array = cupy.fromDlpack(dlpack)

    return array

def tf_to_cp_decorator(num_args):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            if len(args) < num_args:
                raise ValueError("Not enough arguments provided")
            
            print("Here 1")
            tf_tensors = args[:num_args]
            non_tf_args = args[num_args:]

            device_nums = {int(tf_tensor.device.split(':')[-1]) for tf_tensor in tf_tensors}
            print(device_nums)

            if len(device_nums) > 1:
                raise ValueError("All tensors must be on the same device")

            cupy.cuda.Device(device_nums.pop()).use()
            
            print("Here 1")

            cp_arrays = [tensor_to_cupy(tf_tensor) for tf_tensor in tf_tensors]
            
            print("Here 3")

            result_cp_array = func(*cp_arrays, *non_tf_args, **kwargs)
            
            print("Here 4")
            
            result_cp_array = cupy.ascontiguousarray(result_cp_array)

            result_tf_tensor = cupy_to_tensor(result_cp_array)

            return result_tf_tensor

        return wrapper
    return actual_decorator