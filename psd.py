import tensorflow as tf
import tensorflow_probability  as tfp

@tf.function 
def fftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    results = tf.range(0, n // 2 + 1, dtype=tf.float32)  # Note the +1 here
    return results * val

@tf.function 
def calculate_psd(
        signal, 
        nperseg, 
        noverlap=None, 
        sample_rate_hertz = 1.0, 
        mode="mean"
    ):
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Step 1: Split the signal into overlapping segments
    signal_shape = tf.shape(signal)
    step = nperseg - noverlap
    frames = tf.signal.frame(signal, frame_length=nperseg, frame_step=step)
        
    # Step 2: Apply a window function to each segment
    # Hanning window is used here, but other windows can be applied as well
    window = tf.signal.hann_window(nperseg, dtype = tf.float32)
    windowed_frames = frames * window
    
    # Step 3: Compute the periodogram (scaled, absolute value of FFT) for each segment
    periodograms = tf.abs(tf.signal.rfft(windowed_frames)) ** 2 / tf.reduce_sum(window ** 2)
    
    # Step 4: Compute the median or mean of the periodograms based on the median_mode
    if mode == "median":
        pxx = tfp.stats.percentile(periodograms, 50.0, axis=-2)
    elif mode == "mean":
        pxx = tf.reduce_mean(periodograms, axis=-2)
    else:
        raise "Mode not supported"
    
    # Step 5: Compute the frequencies corresponding to the power spectrum values
    freqs = fftfreq(nperseg, d=1.0/sample_rate_hertz)
    
    return freqs, (2.0*pxx / sample_rate_hertz)