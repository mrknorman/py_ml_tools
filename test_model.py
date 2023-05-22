from .dataset import get_ifo_data_generator, get_ifo_data, O3
from .model import ModelBuilder, HyperParameter, DenseLayer, ConvLayer, Population, randomizeLayer
from .setup import load_label_datasets, setup_cuda, get_element_shape

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import mixed_precision, layers
from tensorflow.keras import backend as K
from tensorflow.data.experimental import AutoShardPolicy
import tensorflow as tf
import numpy as np
import os

def load_cbc_datasets(data_directory_path, num_to_load):
    dataset_prefix = f"{data_directory_path}/datasets/cbc"
    dataset_paths = [f"{dataset_prefix}_{i}_v" for i in range(num_to_load)]
    
    # Check for existing dataset paths
    existing_paths = []
    for path in dataset_paths:
        if os.path.exists(path):
            existing_paths.append(path)
        else:
            print(f"Warning: {path} does not exist.")
    
    # Load the datasets with existing paths
    dataset = load_label_datasets(existing_paths, 1)
    
    return dataset

def gaussian_noise_generator(num_samples=8192):
    while True:
        noise = tf.random.normal([num_samples], dtype=tf.float16)
        constant = tf.constant(0.0, dtype=tf.float32)
        yield noise, constant

if __name__ == "__main__":
    # Load datasets:
    strategy = setup_cuda("6", verbose = True)
        
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    skywarp_data_directory = "./skywarp_data/"
    batch_size = 32
    max_populaton = 10
    
    with strategy.scope():
        
        cbc_ds = load_cbc_datasets(skywarp_data_directory, 1)
        num_samples = get_element_shape(cbc_ds)[0]

         # Create TensorFlow dataset from the generator
        noise_ds = tf.data.Dataset.from_generator(
            generator=lambda: gaussian_noise_generator(num_samples=num_samples),
            output_signature=(
                tf.TensorSpec(shape=(num_samples,), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            )
        )

        cbc_ds = cbc_ds \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .with_options(options) \
            .take(100000) 

        noise_ds = noise_ds \
            .batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .with_options(options)

        balanced_dataset = cbc_ds.concatenate(noise_ds \
            .take(tf.data.experimental.cardinality(cbc_ds).numpy())) \
            .prefetch(tf.data.experimental.AUTOTUNE) \
            .with_options(options)
        
        optimizer = \
            HyperParameter(
                "adam", 
                {"type" : "this", "values" : ['adam']}
            )
        
        max_num_inital_layers = 10
        num_layers = \
            HyperParameter(
                10, 
                {"type" : "int_range", "values" : [1, max_num_inital_layers]}
            )
        batch_size = \
            HyperParameter(
                batch_size, 
                {"type" : "this", "values" : [batch_size]}
            )
        
        activations = \
            HyperParameter(
                "relu", 
                {"type" : "list", "values" : ['relu', 'elu', 'sigmoid', 'tanh']}
            )
        d_units = \
            HyperParameter(
                8, 
                {"type" : "power_2_range", "values" : [16, 256]}
            )
        filters = \
            HyperParameter(
                8, 
                {"type" : "power_2_range", "values" : [16, 256]}
            )
        kernel_size = \
            HyperParameter(
                8, 
                {"type" : "int_range", "values" : [1, 7]}
            )
        strides = \
            HyperParameter(
                8, 
                {"type" : "int_range", "values" : [1, 7]}
            )
        
        param_limits = {
            "Dense" : DenseLayer(d_units,  activations),
            "Convolutional":  ConvLayer(filters, kernel_size, activations, strides)
        }
        
        
        genome_template = {
            'base' : {
                'optimizer'  : optimizer,
                'num_layers' : num_layers,
                'batch_size' : batch_size
            },
            'layers' : [
                (["Dense", "Convolutional"], param_limits) \
                for i in range(max_num_inital_layers)
            ]
        }
        
        population = Population(10, 15, genome_template, num_samples, 2)
        population.train_population(1, balanced_dataset)
        
