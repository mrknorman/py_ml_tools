import cusignal

import cupy
import cupyx

def planck(N: int, nleft: int, nright: int) -> cupy.ndarray:
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
    window : cupy.ndarray
        A window of length `N` with a Planck taper applied.
    """
    # Creating left and right ranges
    left = cupy.arange(nleft)
    right = cupy.arange(nright) - nright + 1
    
    # Apply the Planck-taper function to left and right ranges
    taper_left = 1 / (cupy.exp(-left/(nleft-1)) + 1)
    taper_right = 1 / (cupy.exp(-right/(nright-1)) + 1)
    
    # Combine the left taper, a flat middle segment, and the right taper
    window = cupy.concatenate((taper_left, cupy.ones(N-nleft-nright), taper_right[::-1]))
    
    return window

def truncate_transfer(transfer: cupy.ndarray, ncorner: int = None) -> cupy.ndarray:
    """
    Smoothly zero the edges of a frequency domain transfer function.
    
    Parameters
    ----------
    transfer : cupy.ndarray
        The transfer function to truncate.
    ncorner : int, optional
        The number of extra samples to zero off at low frequency.
        
    Returns
    -------
    transfer : cupy.ndarray
        The truncated transfer function.
    """
    nsamp = transfer.shape[-1]
    ncorner = ncorner if ncorner else 0
    
    # Validate that ncorner is within the range of the array size
    if ncorner >= nsamp:
        raise ValueError("ncorner must be less than the size of the transfer array")
        
    plank = planck(nsamp-ncorner, nleft=5, nright=5)
        
    transfer[:ncorner] = 0
    transfer[ncorner:nsamp] *= planck(nsamp-ncorner, nleft=5, nright=5)
    
    return transfer

def truncate_impulse(impulse: cupy.ndarray, ntaps: int, window: str = 'hann') -> cupy.ndarray:
    """
    Smoothly truncate a time domain impulse response.
    
    Parameters
    ----------
    impulse : cupy.ndarray
        The impulse response to truncate.
    ntaps : int
        Number of taps in the final filter, must be an even number.
    window : str, optional
        Window function to truncate with, default is 'hann'.
        
    Returns
    -------
    impulse: cupy.ndarray
        The truncated impulse response.
    """
    
    # Ensure ntaps does not exceed the size of the impulse response
    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")
    
    trunc_start = int(ntaps / 2)
    trunc_stop = impulse.shape[-1] - trunc_start
        
    window = cusignal.windows.windows.get_window(window, ntaps)
    
    if impulse.ndim == 1:
        impulse[:trunc_start] *= window[trunc_start:ntaps]
        impulse[trunc_stop:] *= window[:trunc_start]
        impulse[trunc_start:trunc_stop] = 0
    elif impulse.ndim == 2: 
        impulse[:,:trunc_start] *= window[trunc_start:ntaps]
        impulse[:,trunc_stop:] *= window[:trunc_start]
        impulse[:,trunc_start:trunc_stop] = 0
        

    return impulse

def fir_from_transfer(transfer: cupy.ndarray, ntaps: int, window: str = 'hann', ncorner: int = 0) -> cupy.ndarray:
    """Design a Type II FIR filter given an arbitrary transfer function
    Parameters
    ----------
    transfer : `cupy.ndarray`
        transfer function to start from, must have at least ten samples
    ntaps : `int`
        number of taps in the final filter, must be an even number
    window : `str`, `cupy.ndarray`, optional
        window function to truncate with, default: ``'hann'``
    ncorner : `int`, optional
        number of extra samples to zero off at low frequency, default: `None`
    Returns
    -------
    impulse : `cupy.ndarray`
        A time domain FIR filter of length `ntaps`
    """
    if ntaps % 2 != 0:
        raise ValueError("ntaps must be an even number")
    
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    impulse = cupyx.scipy.fftpack.irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    
    impulse = cupy.roll(impulse, int(ntaps/2 - 1))[: ntaps]
    return impulse

def convolve2d(timeseries: cupy.ndarray, fir: cupy.ndarray, window: str = 'hann') -> cupy.ndarray:
    """
    Perform convolution between the timeseries and the finite impulse response filter.
    
    Parameters
    ----------
    timeseries : cupy.ndarray
        The time series data to convolve.
    fir : cupy.ndarray
        The finite impulse response filter.
    window : str, optional
        Window function to use, default is 'hann'.
        
    Returns
    -------
    conv : cupy.ndarray
        The convolved time series.
    """
    pad = int(cupy.ceil(fir.shape[-1]/2))
    
    # Optimizing FFT size to power of 2 for efficiency
    nfft = min(8*fir.shape[-1], timeseries.shape[-1])

    window = cusignal.windows.get_window(window, fir.shape[-1])

    timeseries[:, :pad] *= window[:pad]
    timeseries[:, -pad:] *= window[-pad:]

    conv = cupy.zeros_like(timeseries)
    if nfft >= timeseries.shape[-1]/2:
        conv = cusignal.convolve(timeseries, fir, mode='same', method = 'fft')
    else:
        nstep = nfft - 2*pad
        conv[:, :nfft-pad] = cusignal.convolve(timeseries[:, :nfft], fir, mode='same', method = 'fft')[:, :nfft-pad]

        k = nfft - pad
        while k < timeseries.shape[-1] - nfft + pad:
            yk = cusignal.convolve(timeseries[:, k-pad:k+nstep+pad], fir, mode='same', method = 'fft')
            conv[:, k:k+yk.shape[-1]-2*pad] = yk[:, pad:-pad]
            k += nstep

        conv[:, -nfft+pad:] = cusignal.convolve(timeseries[:, -nfft:], fir, mode='same', method = 'fft')[:, -nfft+pad:]

    return conv

def convolve(timeseries: cupy.ndarray, fir: cupy.ndarray, window: str = 'hann') -> cupy.ndarray:
    """
    Perform convolution between the timeseries and the finite impulse response filter.
    
    Parameters
    ----------
    timeseries : cupy.ndarray
        The time series data to convolve.
    fir : cupy.ndarray
        The finite impulse response filter.
    window : str, optional
        Window function to use, default is 'hann'.
        
    Returns
    -------
    conv : cupy.ndarray
        The convolved time series.
    """
    pad = int(cupy.ceil(fir.size/2))
    
    # Optimizing FFT size to power of 2 for efficiency
    nfft = min(8*fir.shape[-1], timeseries.shape[-1])

    window = cusignal.windows.get_window(window, fir.size)

    timeseries[:pad] *= window[:pad]
    timeseries[-pad:] *= window[-pad:]

    if nfft >= timeseries.size/2:
        conv = cusignal.convolve(timeseries, fir, mode='same', method = 'fft')
    else:
        nstep = nfft - 2*pad
        conv = cupy.zeros(timeseries.size)
        conv[:nfft-pad] = cusignal.convolve(timeseries[:nfft], fir, mode='same', method = 'fft')[:nfft-pad]

        k = nfft - pad
        while k < timeseries.size - nfft + pad:
            yk = cusignal.convolve(timeseries[k-pad:k+nstep+pad], fir, mode='same', method = 'fft')
            conv[k:k+yk.size-2*pad] = yk[pad:-pad]
            k += nstep

        conv[-nfft+pad:] = cusignal.convolve(timeseries[-nfft:], fir, mode='same', method = 'fft')[-nfft+pad:]

    return conv

def interpolate(new_x, x, y):
    if y.ndim == 1:
        result = cupy.interp(new_x, x, y)
    elif y.ndim == 2:
        # Initialize an empty array for the result
        result = cupy.empty((y.shape[0], len(new_x)))

        # Apply cupy.interp to each sub-array
        for i in range(y.shape[0]):
            result[i] = cupy.interp(new_x, x, y[i])
        
    return result

def whiten(
    timeseries: cupy.ndarray, 
    background: cupy.ndarray,
    sample_rate: float, 
    fftlength: int = 4, 
    overlap: int = 2,
    highpass: float = None,
    detrend ='constant',
    fduration: int = 1.0,
    window: str = "hann"
    ) -> cupy.ndarray:
    
    """
    Whiten a timeseries using the given parameters.
    
    Parameters
    ----------
    timeseries : cupy.ndarray
        The time series data to whiten.
    background : cupy.ndarray
        The time series to use to calculate the asd.
    sample_rate : float
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
    out : cupy.ndarray
        The whitened time series.
    """
    dt = 1 / sample_rate
            
    freqs, psd = \
        cusignal.spectral_analysis.spectral.csd(
            background, 
            background, 
            fs=sample_rate, 
            window=window, 
            nperseg=int(sample_rate*fftlength), 
            noverlap=int(sample_rate*overlap), 
            average='median')
    
    asd = cupy.sqrt(psd)
    
    df = 1.0 / (timeseries.shape[-1] / sample_rate)
        
    fsamples = cupy.arange(0, timeseries.shape[-1]//2+1) * df
    freqs = cupy.asarray(freqs)
    
    if (asd.shape[-1] != timeseries.shape[-1]//2+1):
        asd = interpolate(fsamples, freqs, asd) 
    
    ncorner = int(highpass / df) if highpass else 0
    ntaps = int(fduration * sample_rate)
    transfer = 1.0 / asd
            
    tdw = fir_from_transfer(transfer, ntaps, window=window, ncorner=ncorner)
    timeseries = cusignal.filtering.detrend(timeseries, type=detrend, overwrite_data=True)
    
    if timeseries.ndim == 1:
        out = convolve(timeseries, tdw, window=window)
    elif timeseries.ndim == 2:
        out = convolve2d(timeseries, tdw, window=window)
    
    return out * cupy.sqrt(dt)