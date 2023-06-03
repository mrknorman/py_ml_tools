from .dataset import get_ifo_data, O3
from typing import Dict, Tuple, Optional

import tensorflow as tf
import numpy as np

from tensorflow.data.experimental import AutoShardPolicy
from .dataset import get_ifo_data_generator, O3

#REMOVE WHEN STRAIN IS RENAMED TO ONSOURCE IN MODEL:
def getInput(element):
    return element['onsource']

@tf.function 
def nan_to_zero(tensor):
    return tf.where(tf.math.is_nan(tensor), 0.0, tensor)

def calculate_far_scores(
    model: tf.keras.Model,
    sample_rate_hertz: int,
    onsource_duration_seconds: float,
    ifo: str,
    time_interval = O3,
    num_examples_per_batch: int = 32,
    seed: int = 100,
    num_examples: int = int(1E5)
) -> np.ndarray:
    """
    Calculate the False Alarm Rate (FAR) scores for a given model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    sample_rate_hertz : int
        The sample rate in hertz.
    onsource_duration_seconds : float
        The duration of onsource in seconds.
    ifo : str
        The name of the Interferometric Gravitational-Wave Observatory.
    time_interval : str, optional
        The time interval for the IFO data generator, default is 'O3'.
    num_examples_per_batch : int, optional
        The number of examples per batch, default is 32.
    seed : int, optional
        The seed used for random number generation, default is 100.
    num_examples : float, optional
        The total number of examples to be used, default is 1E5.

    Returns
    -------
    far_scores : np.ndarray
        The calculated FAR scores.

    """
    
    num_examples = int(num_examples)
    
    # Setting options for data distribution
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    
    # Creating the noise dataset
    noise_ds = get_ifo_data_generator(
        time_interval = time_interval,
        data_labels = ["noise", "glitches"],
        ifo = ifo,
        sample_rate_hertz = sample_rate_hertz,
        onsource_duration_seconds = onsource_duration_seconds,
        max_segment_size = 3600,
        num_examples_per_batch = num_examples_per_batch,
        order = "random",
        seed = seed,
        apply_whitening = True,
        return_keys = ["onsource"], 
    ).map(getInput, num_parallel_calls=tf.data.AUTOTUNE)\
    .prefetch(tf.data.experimental.AUTOTUNE).with_options(options)
    
    # Calculating the number of steps for model prediction
    num_steps = int(num_examples // num_examples_per_batch)
    noise_ds = noise_ds.take(num_examples)
    
    # Predicting the scores and getting the second column ([:, 1])
    far_scores = model.predict(noise_ds, steps = num_steps, verbose=2)
    far_scores = nan_to_zero(far_scores)
    print(far_scores[:, 1])
    
    return far_scores[:, 1]

def calculate_far_score_thresholds(
    far_scores: np.ndarray, 
    onsource_duration_seconds: float,
    fars: np.ndarray  # Changed this to np.ndarray to leverage vectorized operations
) -> Dict[float, Tuple[float, float]]:
    """
    Calculate the score thresholds for False Alarm Rate (FAR).

    Parameters
    ----------
    far_scores : np.ndarray
        The FAR scores calculated previously.
    onsource_duration_seconds : float
        The duration of onsource in seconds.
    model_data : np.ndarray
        The data used to train the model.
    fars : np.ndarray
        Array of false alarm rates.

    Returns
    -------
    score_thresholds : Dict[float, Tuple[float, float]]
        Dictionary of false alarm rates and their corresponding score thresholds.

    """
    # Sorting the FAR scores in descending order
    far_scores = np.sort(far_scores)[::-1]

    # Calculating the total number of seconds
    total_num_seconds = len(far_scores) * onsource_duration_seconds

    # Creating the far axis
    far_axis = (np.arange(total_num_seconds) + 1) / total_num_seconds
    
    # Find the indexes of the closest FAR values in the far_axis
    idxs = np.abs(np.subtract.outer(far_axis, fars)).argmin(axis=0)
    # Find the indexes of the closest scores in the far_scores
    idxs = np.abs(np.subtract.outer(far_scores, far_scores[idxs])).argmin(axis=0)

    # Build the score thresholds dictionary
    score_thresholds = {far: (far, far_scores[idx]) for far, idx in zip(fars, idxs)}

    # If any score is 1, set the corresponding threshold to 1.1
    for far, (_, score) in score_thresholds.items():
        if score == 1:
            score_thresholds[far] = (far, 1.1)

    return score_thresholds

def validate_far(
    model: tf.keras.Model,
    sample_rate_hertz: int,
    onsource_duration_seconds: float,
    ifo: str,
    time_interval: str = O3,
    num_examples_per_batch: int = 32,
    seed: int = 100,
    num_examples: float = 1E5,
    validation_fars: Optional[np.ndarray] = None
    ):
    """
    Validate the False Alarm Rate (FAR) of the model.

    Parameters
    ----------
    model : tf.keras.Model
        The model used to predict FAR scores.
    sample_rate_hertz : int
        The sample rate in hertz.
    onsource_duration_seconds : float
        The duration of onsource in seconds.
    ifo : str
        The name of the Interferometric Gravitational-Wave Observatory.
    time_interval : str, optional
        The time interval for the IFO data generator, default is 'O3'.
    num_examples_per_batch : int, optional
        The number of examples per batch, default is 32.
    seed : int, optional
        The seed used for random number generation, default is 100.
    num_examples : float, optional
        The total number of examples to be used, default is 1E5.
    validation_fars : np.ndarray, optional
        The array of false alarm rates for validation, default is logspace from 
        1/(num_examples*onsource_duration_seconds) to 1.

    """
    # If validation_fars not provided, generate it from num_examples and 
    # onsource_duration_seconds
    if validation_fars is None:
        start_power = np.log10(1/(num_examples*onsource_duration_seconds))
         # Calculate number of points to cover all integer exponents
        num_points = int(np.ceil(1 - start_power)) + 1 
        validation_fars = np.logspace(start_power, 0, num=num_points)
    
    far_scores = calculate_far_scores(
        model,
        sample_rate_hertz,
        onsource_duration_seconds,
        ifo,
        time_interval,
        num_examples_per_batch,
        seed,
        num_examples
    )
    
    score_thresholds = calculate_far_score_thresholds(
        far_scores, 
        onsource_duration_seconds,
        validation_fars
    )

    return score_thresholds

    