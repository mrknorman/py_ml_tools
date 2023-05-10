import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy import signal

def whiten_tf(timeseries, sample_rate, fftlength=4, overlap=2):
    # Compute the ASD
    dt = 1 / sample_rate
    asd = tf.sqrt(timeseries_psd(timeseries, sample_rate, fftlength, overlap=overlap))
    
    # Design whitening filter
    ntaps = int((2 * sample_rate))
    tdw = fir_from_transfer_tf(tf.cast(1, dtype=tf.complex64) / tf.cast(asd, dtype=tf.complex64), ntaps)
    
    # Condition the input data and apply the whitening filter
    in_ = detrend_constant_tf(timeseries)
    out = convolve_tf(in_, tdw)
    return out * tf.sqrt(2 * dt)

def fir_from_transfer_tf(transfer, ntaps, ncorner=0):
    transfer = tf.convert_to_tensor(transfer, dtype=tf.complex64)

    # Design an FIR filter using the inverse FFT
    half_ntaps = ntaps // 2
    inverse_fft = tf.signal.irfft(transfer, fft_length=[ntaps])[:half_ntaps]
    
    # Apply the Hann window function
    win = tf.signal.hann_window(ntaps)

    # Apply the window and shift the filter taps
    fir = tf.roll(inverse_fft * win[:half_ntaps], shift=half_ntaps, axis=0)

    # Apply the high-pass filter, if requested
    if ncorner > 0:
        fir[ncorner:] = 0.0

    return fir

def timeseries_psd(timeseries, sample_rate, fftlength, overlap):
    # Convert input to tensor
    timeseries = tf.convert_to_tensor(timeseries, dtype=tf.float32)
    
    # Create a Hann window tensor
    win = tf.signal.hann_window(fftlength * sample_rate)

    # Calculate the step size and number of segments
    step_size = int((fftlength - overlap) * sample_rate)
    num_segments = (timeseries.shape[0] - win.shape[0]) // step_size + 1

    # Reshape the time series into segments using tf.signal.frame
    timeseries_segments = tf.signal.frame(timeseries, frame_length=win.shape[0], frame_step=step_size)

    # Apply the window function to the time series segments
    timeseries_windowed = timeseries_segments * win

    # Compute the short-time Fourier transform (STFT)
    stft = tf.signal.stft(timeseries_windowed, frame_length=fftlength * sample_rate,
                          frame_step=(fftlength - overlap) * sample_rate,
                          fft_length=fftlength * sample_rate)

    # Compute the power spectral density (PSD)
    psd = tf.math.real(tf.math.conj(stft) * stft)

    # Average the PSD based on the median method
    psd = tfp.stats.percentile(psd, 50.0, axis=-1, keepdims=False)

    return psd

def detrend_constant_tf(timeseries):
    mean = tf.reduce_mean(timeseries)
    detrended = timeseries - mean
    return detrended

def convolve_tf(timeseries, fir):
    pad = tf.cast(tf.math.ceil(fir.shape[0] / 2), dtype=tf.int32)
    nfft = tf.minimum(8 * fir.shape[0], timeseries.shape[0])

    # Condition the input data
    timeseries = tf.identity(timeseries)

    timeseries = tf.pad(timeseries, [[pad, pad]], mode='REFLECT')

    timeseries[:pad] *= tf.signal.hann_window(fir.shape[0])[:pad]
    timeseries[-pad:] *= tf.signal.hann_window(fir.shape[0])[-pad:]

    # Perform the convolution in the frequency domain
    freq_ts = tf.signal.rfft(timeseries, fft_length=[nfft])
    freq_fir = tf.signal.rfft(fir, fft_length=[nfft])

    conv_freq = freq_ts * freq_fir
    conv = tf.signal.irfft(conv_freq)

    # Remove the padding
    conv = conv[pad:-pad]

    return conv

def test_whiten_functions(sample_rate=4096, duration=8, fftlength=4, overlap=2, tol=1e-6):
    # Generate random data
    np.random.seed(42)
    data = np.random.normal(0, 1, sample_rate * duration)

    # Create a GWpy TimeSeries object
    ts = TimeSeries(data, sample_rate=sample_rate)

    # Whiten the data using the original function
    whitened_ts = ts.whiten(fftlength=fftlength, overlap=overlap).value

    # Whiten the data using the TensorFlow-based function
    whitened_tf = whiten_tf(data, sample_rate, fftlength=fftlength, overlap=overlap).numpy()

    # Compare the results
    assert np.allclose(whitened_ts, whitened_tf, atol=tol), "The outputs of the two functions do not match."

# Run the test function
test_whiten_functions()