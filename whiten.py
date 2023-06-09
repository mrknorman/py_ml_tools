import tensorflow as tf
from .psd import calculate_psd
import tensorflow_probability  as tfp
import tensorflow.signal as tfs

@tf.function 
def planck(N: int, nleft: int, nright: int) -> tf.Tensor:
    """
    Create a Planck-taper window.
    
    Parameters
    ----------
    N : int
        The total number of samples in the window.
    nleft : int
        The number of samples in the left taper segment.
    nright : int
        The number of samples in the right taper segment.
        
    Returns
    -------
    window : tf.Tensor
        A window of length `N` with a Planck taper applied.
    """
    # Creating left and right ranges
    left = tf.range(nleft, dtype=tf.float32)
    right = tf.range(nright, dtype=tf.float32) - nright + 1
    
    # Apply the Planck-taper function to left and right ranges
    taper_left = 1 / (tf.exp(-left/(nleft-1)) + 1)
    taper_right = 1 / (tf.exp(-right/(nright-1)) + 1)
    
    # Combine the left taper, a flat middle segment, and the right taper
    window = tf.concat([
        taper_left, 
        tf.ones(N-nleft-nright), 
        tf.reverse(taper_right, axis=[0])
    ], axis=0)
    
    return window

@tf.function 
def truncate_transfer(
    transfer: tf.Tensor,
    ncorner: int = None
    ) -> tf.Tensor:
    """
    Smoothly zero the edges of a frequency domain transfer function.
    
    Parameters
    ----------
    transfer : tf.Tensor
        The transfer function to truncate.
    ncorner : int, optional
        The number of extra samples to zero off at low frequency.
        
    Returns
    -------
    transfer : tf.Tensor
        The truncated transfer function.
    """
    nsamp = transfer.shape[-1]
    ncorner = ncorner if ncorner else 0
    
    # Validate that ncorner is within the range of the array size
    if ncorner >= nsamp:
        raise ValueError(
            "ncorner must be less than the size of the transfer array"
        )
        
    plank = planck(nsamp-ncorner, nleft=5, nright=5)
    
    transfer_zeros = tf.zeros_like(transfer[:,:ncorner])
    transfer_mod = tf.multiply(transfer[:,ncorner:nsamp], plank)
    new_transfer = tf.concat([transfer_zeros, transfer_mod], axis=-1)
    
    return new_transfer

@tf.function 
def truncate_impulse(
    impulse: tf.Tensor, 
    ntaps: int, 
    window: str = 'hann'
    ) -> tf.Tensor:
    """
    Smoothly truncate a time domain impulse response.
    
    Parameters
    ----------
    impulse : tf.Tensor
        The impulse response to truncate.
    ntaps : int
        Number of taps in the final filter, must be an even number.
    window : str, optional
        Window function to truncate with, default is 'hann'.
        
    Returns
    -------
    impulse: tf.Tensor
        The truncated impulse response.
    """
    
    # Ensure ntaps does not exceed the size of the impulse response
    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")
    
    trunc_start = int(ntaps / 2)
    trunc_stop = impulse.shape[-1] - trunc_start
        
    if window == 'hann':
        window = tfs.hann_window(ntaps)
    # Extend this section with more cases if more window functions are required.
    else:
        raise ValueError(f"Window function {window} not supported")
    
    impulse_start = impulse[:,:trunc_start] * window[trunc_start:ntaps]
    impulse_stop = impulse[:,trunc_stop:] * window[:trunc_start]
    impulse_middle = tf.zeros_like(impulse[:,trunc_start:trunc_stop])
    
    new_impulse = tf.concat([impulse_start, impulse_middle, impulse_stop], axis=-1)

    return new_impulse


@tf.function 
def fir_from_transfer(
    transfer: tf.Tensor, 
    ntaps: int, 
    window: str = 'hann', 
    ncorner: int = 0
    ) -> tf.Tensor:
    """
    Design a Type II FIR filter given an arbitrary transfer function

    Parameters
    ----------
    transfer : tf.Tensor
        transfer function to start from, must have at least ten samples
    ntaps : int
        number of taps in the final filter, must be an even number
    window : str, tf.Tensor, optional
        window function to truncate with, default: 'hann'
    ncorner : int, optional
        number of extra samples to zero off at low frequency, default: 0

    Returns
    -------
    impulse : tf.Tensor
        A time domain FIR filter of length ntaps
    """

    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")
    
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    impulse = tf.signal.irfft(tf.cast(transfer, dtype=tf.complex64)) # the equivalent of irfft
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    
    impulse = tf.roll(impulse, shift=int(ntaps/2 - 1), axis=-1)[:,: ntaps]
    return impulse

@tf.function 
def fftconvolve_(in1, in2, mode="full"):
    """Convolve two N-dimensional arrays using FFT.

    This function works similarly to the fftconvolve function you provided,
    but uses TensorFlow's signal processing API.
    """

    in1 = tf.constant(in1)
    in2 = tf.constant(in2)

    if in1.shape.ndims != in2.shape.ndims:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif tf.size(in1) == 0 or tf.size(in2) == 0:  # empty arrays
        return tf.constant([])

    s1 = tf.shape(in1)
    s2 = tf.shape(in2)

    complex_result = (tf.dtypes.as_dtype(in1.dtype).is_complex or
                      tf.dtypes.as_dtype(in2.dtype).is_complex)
    shape = tf.maximum(s1, s2)
    shape = s1 + s2 - 1

    # Check that input sizes are compatible with 'valid' mode
    if mode == 'valid' and tf.reduce_any(s1 < s2):
        # Convolution is commutative; order doesn't have any effect on output
        in1, s1, in2, s2 = in2, s2, in1, s1

    if not complex_result:
        sp1 = tf.signal.rfft(in1)
        sp2 = tf.signal.rfft(in2)
        ret = tf.signal.irfft(sp1 * sp2)
    else:
        sp1 = tf.signal.fft(in1)
        sp2 = tf.signal.fft(in2)
        ret = tf.signal.ifft(sp1 * sp2)

    if mode == "full":
        return ret
    elif mode == "same":
        start = s1 // 2
        return tf.slice(ret, [start], [s1])
    elif mode == "valid":
        start = s2 - 1
        return tf.slice(ret, [start], [s1 - s2 + 1])
    else:
        raise ValueError(
            "acceptable mode flags are 'valid', 'same', or 'full'"
        )

@tf.function 
def _centered(arr, newsize):
    # Ensure correct dimensionality
    if len(arr.shape) == 1:
        arr = tf.expand_dims(arr, 0)
    # Calculate start and end indices
    start_ind = (arr.shape[-1] - newsize) // 2
    end_ind = start_ind + newsize
    return arr[..., start_ind:end_ind]

@tf.function 
def fftconvolve(in1, in2, mode="full"):
    # Extract shapes
    s1 = tf.shape(in1)[-1]
    s2 = tf.shape(in2)[-1]
    shape = s1 + s2 - 1

    # Compute convolution in Fourier space
    sp1 = tf.signal.rfft(in1, [shape])
    sp2 = tf.signal.rfft(in2, [shape])
    ret = tf.signal.irfft(sp1 * sp2, [shape])

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered(ret, s1)
    elif mode == "valid":
        cropped = _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

    return cropped

@tf.function 
def convolve(
    timeseries: tf.Tensor, 
    fir: tf.Tensor, 
    window: str = 'hann'
    ) -> tf.Tensor:
    """
    Perform convolution between the timeseries and the finite impulse response 
    filter.
    
    Parameters
    ----------
    timeseries : tf.Tensor
        The time series data to convolve.
    fir : tf.Tensor
        The finite impulse response filter.
    window : str, optional
        Window function to use, default is 'hann'.
        
    Returns
    -------
    conv : tf.Tensor
        The convolved time series.
    """
    pad = int(tf.math.ceil(fir.shape[-1]/2))
    
    # Optimizing FFT size to power of 2 for efficiency
    nfft = min(8*fir.shape[-1], timeseries.shape[-1])

    if window == 'hann':
        window = tf.signal.hann_window(fir.shape[-1])
    # Extend this section with more cases if more window functions are required.
    else:
        raise ValueError(f"Window function {window} not supported")

    timeseries_new_front = timeseries[:, :pad] * window[:pad]
    timeseries_new_back = timeseries[:, -pad:] * window[-pad:]
    timeseries_new_middle = timeseries[:, pad:-pad]

    timeseries_new = tf.concat([
        timeseries_new_front, 
        timeseries_new_middle, 
        timeseries_new_back
    ], axis=1)

    conv = tf.zeros_like(timeseries_new)
    if nfft >= timeseries_new.shape[-1]/2:
        conv = fftconvolve(timeseries_new, fir, mode='same')
    else:
        nstep = nfft - 2*pad
        conv[:, :nfft-pad] = fftconvolve(
            timeseries_new[:, :nfft], 
            fir, 
            mode='same'
        )[:, :nfft-pad]

        k = nfft - pad
        while k < timeseries_new.shape[-1] - nfft + pad:
            yk = fftconvolve(
                timeseries_new[:, k-pad:k+nstep+pad], 
                fir,
                mode='same'
            )
            conv[:, k:k+yk.shape[-1]-2*pad] = yk[:, pad:-pad]
            k += nstep

        conv[:, -nfft+pad:] = fftconvolve(
            timeseries_new[:, -nfft:], fir, mode='same'
        )[:, -nfft+pad:]

    return conv

@tf.function 
def whiten(
    timeseries: tf.Tensor, 
    background: tf.Tensor,
    sample_rate_hertz: float, 
    fftlength: int = 4, 
    overlap: int = 2,
    highpass: float = None,
    detrend: str ='constant',
    fduration: int = 2.0,
    window: str = "hann"
    ) -> tf.Tensor:
    """
    Whiten a timeseries using the given parameters.
    
    Parameters
    ----------
    timeseries : tf.Tensor
        The time series data to whiten.
    background : tf.Tensor
        The time series to use to calculate the asd.
    sample_rate_hertz : float
        The sample rate of the time series data.
    fftlength : int, optional
        Length of the FFT window, default is 4.
    overlap : int, optional
        Overlap of the FFT windows, default is 2.
    highpass : float, optional
        Highpass frequency, default is None.
    fduration : int, optional
        Duration of the filter in seconds, default is 2.
    window : str, optional
        Window function to use, default is 'hann'.
        
    Returns
    -------
    out : tf.Tensor
        The whitened time series.
    """

    # Check if input is 1D or 2D
    is_1d = len(timeseries.shape) == 1
    if is_1d:
        # If 1D, add an extra dimension
        timeseries = tf.expand_dims(timeseries, axis=0)
        background = tf.expand_dims(background, axis=0)

    dt = 1 / sample_rate_hertz
    
    freqs, psd = calculate_psd(
        background, 
        nperseg=int(sample_rate_hertz*fftlength), 
        noverlap=int(sample_rate_hertz*overlap), 
        sample_rate_hertz=sample_rate_hertz
    )
    asd = tf.sqrt(psd)
    
    df = 1.0 / (timeseries.shape[-1] / sample_rate_hertz)
    fsamples = tf.range(0, timeseries.shape[-1]//2+1, dtype=tf.float32) * df
    freqs = tf.cast(freqs, tf.float32)
    
    asd = \
        tfp.math.interp_regular_1d_grid(
            fsamples, 
            freqs[0], 
            freqs[-1], 
            asd, 
            axis=-1
        )

    ncorner = int(highpass / df) if highpass else 0
    ntaps = int(fduration * sample_rate_hertz)
    transfer = 1.0 / asd

    tdw = fir_from_transfer(transfer, ntaps, window=window, ncorner=ncorner)
    out = convolve(timeseries, tdw)

    # If input was 1D, return 1D
    if is_1d:
        out = out[0]

    return out * tf.sqrt(2.0 * dt)