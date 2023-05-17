import tensorflow as tf
import os

def setup_cuda(
    device_num : str,
    verbose : bool = False
    ):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    strategy = tf.distribute.MirroredStrategy()
    
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