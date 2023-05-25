import tensorflow as tf
from .psd import calculate_psd
import tensorflow_probability as tfp
import tensorflow.signal as tfs
import numpy as np

@tf.function 
def calculate_snr(
    injection: tf.Tensor, 
    background: tf.Tensor,
    window_duration_seconds: float, 
    sample_rate_hertz: float, 
    fft_duration_seconds: float = 4.0, 
    overlap_duration_seconds: float = 2.0,
    lower_frequency_cutoff: float = 20.0,
    ) -> tf.Tensor:
    """
    Calculate the signal-to-noise ratio (SNR) of a given signal.

    Parameters:
    injection : tf.Tensor
        The input signal.
    background : tf.Tensor
        The time series to use to calculate the asd.
    window_duration_seconds : int
        The size of the window over which to compute the SNR.
    sample_rate_hertz : float
        The sampling frequency.
    fft_duration_seconds : int, optional
        Length of the FFT window, default is 4.
    overlap_duration_seconds : int, optional
        Overlap of the FFT windows, default is 2.
    lower_frequency_cutoff : float, optional
        Lower cutoff frequency for the SNR calculation, default is 20Hz.

    Returns:
    SNR : tf.Tensor
        The computed signal-to-noise ratio.
    """
        
    window_num_samples  = int(sample_rate_hertz*window_duration_seconds)
    overlap_num_samples = int(sample_rate_hertz*overlap_duration_seconds)
    fft_num_samples     = int(sample_rate_hertz*fft_duration_seconds)
    
    # Set the frequency integration limits
    upper_frequency_cutoff = int(sample_rate_hertz / 2.0)

    # Calculate and normalize the Fourier transform of the signal
    inj_fft = tf.signal.rfft(injection) / sample_rate_hertz

    # Get rid of DC
    inj_fft_no_dc = inj_fft[1:int(window_num_samples / 2.0) + 1]

    # Compute the square of absolute value
    inj_fft_squared = tf.abs(inj_fft_no_dc*tf.math.conj(inj_fft_no_dc))
    
    # Compute the frequency window for SNR calculation
    start_freq_num_samples = \
        int(window_duration_seconds*lower_frequency_cutoff) - 1
    end_freq_num_samples = \
        int(window_duration_seconds*upper_frequency_cutoff) - 1

    # Calculate PSD of the background noise
    freqs, psd = \
        calculate_psd(
            background, 
            sample_rate_hertz = sample_rate_hertz, 
            nperseg           = fft_num_samples, 
            noverlap          = overlap_num_samples,
            mode="mean"
        )
            
    # Interpolate ASD to match the length of the original signal    
    df = 1.0 / window_duration_seconds
    fsamples = tf.range(0, window_num_samples//2+1, dtype=tf.float32) * df
    freqs = tf.cast(freqs, tf.float32)
    psd_interp = \
        tfp.math.interp_regular_1d_grid(
            fsamples, freqs[0], freqs[-1], psd, axis=-1
        )
    
    # Compute the SNR numerator in the frequency window
    snr_numerator = inj_fft_squared[start_freq_num_samples:end_freq_num_samples]
    
    # Use the interpolated ASD in the frequency window for SNR calculation
    snr_denominator = psd_interp[start_freq_num_samples:end_freq_num_samples]
    
    # Calculate the SNR
    SNR = tf.math.sqrt(
        (4.0 / window_duration_seconds) * tf.reduce_sum(snr_numerator / snr_denominator, axis = -1)
    )

    return SNR

def scale_to_snr(
        injection: tf.Tensor, 
        background: tf.Tensor,
        desired_snr: float,
        window_duration_seconds: float, 
        sample_rate_hertz: float, 
        fft_duration_seconds: float = 4.0, 
        overlap_duration_seconds: float = 2.0,
        lower_frequency_cutoff: float = 20.0
    ):
    
    current_snr = calculate_snr(
        injection, 
        background,
        window_duration_seconds, 
        sample_rate_hertz, 
        fft_duration_seconds = fft_duration_seconds, 
        overlap_duration_seconds = overlap_duration_seconds,
        lower_frequency_cutoff = lower_frequency_cutoff
    )
    
    scale_factor = desired_snr/current_snr
    
    print("Scale factor", scale_factor)
    return injection*scale_factor
    
    