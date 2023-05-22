import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.distribute.distribute_lib import Strategy
from cupy import ndarray as cupy_ndarray
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.data import Dataset

def setup_cuda(device_num: str, verbose: bool = False) -> Strategy:
    """
    Sets up CUDA for TensorFlow. Configures memory growth, logging verbosity, and returns the strategy for distributed computing.

    Args:
        device_num (str): The GPU device number to be made visible for TensorFlow.
        verbose (bool, optional): If True, prints the list of GPU devices. Defaults to False.

    Returns:
        tf.distribute.MirroredStrategy: The TensorFlow MirroredStrategy instance.
    """
    
    # Set up logging to file - this is beneficial in debugging scenarios and for traceability.
    logging.basicConfig(filename='tensorflow_setup.log', level=logging.INFO)
    
    try:
        # Set the device number for CUDA to recognize.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    except Exception as e:
        logging.error(f"Failed to set CUDA_VISIBLE_DEVICES environment variable: {e}")
        raise

    # Confirm TensorFlow and CUDA version compatibility.
    tf_version = tf.__version__
    cuda_version = tf.sysconfig.get_build_info()['cuda_version']
    logging.info(f"TensorFlow version: {tf_version}, CUDA version: {cuda_version}")

    # List all the physical GPUs.
    gpus = tf.config.list_physical_devices('GPU')

    # If any GPU is present.
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs.
            # Enable memory growth for each GPU.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # This needs to be set before initializing GPUs.
            logging.error(f"Failed to set memory growth: GPUs must be initialized first. Error message: {e}")
            raise

    # MirroredStrategy performs synchronous distributed training on multiple GPUs on one machine.
    # It creates one replica of the model on each GPU available.
    strategy = tf.distribute.MirroredStrategy()

    # Set the logging level to ERROR to reduce logging noise.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # If verbose, print the list of GPUs.
    if verbose:
        print(tf.config.list_physical_devices("GPU"))

    # Return the MirroredStrategy instance.
    return strategy

def cupy_to_tensor(cupy_array: cupy_ndarray) -> EagerTensor:
    """
    Converts a CuPy array to a TensorFlow tensor using DLPack.

    Args:
        cupy_array (cupy.ndarray): The CuPy array to be converted.

    Returns:
        tf.Tensor: The converted TensorFlow tensor.
    """
    try:
        # Convert the CuPy array to a DLPack. DLPack is an open standard to tensor data structure sharing across different frameworks.
        dlpack = cupy_array.toDlpack()

        # Convert the DLPack to a TensorFlow tensor.
        tensor = tf.experimental.dlpack.from_dlpack(dlpack)

    except Exception as e:
        logging.error(f"Failed to convert CuPy array to TensorFlow tensor: {e}")
        raise

    return tensor

def load_datasets(paths):
    
    dataset = tf.data.experimental.load(paths[0])
    for path in paths[1:]:
        dataset = dataset.concatenate(dataset)
        
    return dataset

def add_labels(dataset, label):
    dataset_size = dataset.cardinality().numpy()

    labels = Dataset.from_tensor_slices(
        np.full(dataset_size, label, dtype=np.float32))
    
    return Dataset.zip((dataset, labels))
    
def load_label_datasets(paths, label):
    
    dataset = load_datasets(paths)
    return add_labels(dataset, label)

def load_noise_signal_datasets(noise_paths, signal_paths):
    
    noise  = load_label_datasets(noise_paths, 0)
    signal = load_label_datasets(signal_paths, 1)

    return signal.concatenate(noise)

def split_test_train(dataset, fraction):
    dataset_size = dataset.cardinality().numpy()
    test_size = fraction * dataset_size

    dataset = dataset.shuffle(dataset_size)
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)

    return test_dataset, train_dataset

def get_element_shape(dataset):
    for element in dataset.take(1):
        return element[0].shape
    
