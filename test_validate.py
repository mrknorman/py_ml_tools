from .validate import validate_far
from .setup import setup_cuda, find_available_GPUs
import tensorflow as tf
from tensorflow.keras import mixed_precision
import logging

if __name__ == "__main__":
    
     # Load datasets:
        
    gpus = find_available_GPUs(16000, 1)
    strategy = setup_cuda(gpus, verbose = True)
        
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    sample_rate_hertz = 8192.0
    onsource_duration_seconds = 1.0    
        
    with strategy.scope():
        
        logging.info(f"Loading model...")
        model = \
            tf.keras.models.load_model(
                f"./skywarp_data/models/skywarp_conv_regular"
            )
        logging.info("Done.")
    
        threshholds = validate_far(
            model,
            sample_rate_hertz,
            onsource_duration_seconds,
            ifo = 'L1',
            num_examples = 1E4
        )    
        
        print(threshholds)