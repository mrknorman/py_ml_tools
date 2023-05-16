from whiten import whiten
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, Whisker
from gwpy.timeseries import TimeSeries
import numpy as np
import cupy
from cupyx.profiler import benchmark
import timeit

from dataset import get_ifo_data, O3

import scipy

import numpy as np
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure, save
from bokeh.io import output_file

def plot_performance_comparison(cases_dict):
    # Prepare data for plotting
    data = {
        'cases_cpu': [case + "_cpu" for case in cases_dict.keys()],
        'cases_gpu': [case + "_gpu" for case in cases_dict.keys()],
        'cpu_times_mean': [np.mean(case.cpu_times) for case in cases_dict.values()],
        'gpu_times_mean': [np.mean(case.gpu_times) for case in cases_dict.values()],
        'cpu_times_upper': [np.mean(case.cpu_times) + np.std(case.cpu_times) for case in cases_dict.values()],
        'gpu_times_upper': [np.mean(case.gpu_times) + np.std(case.gpu_times) for case in cases_dict.values()],
        'cpu_times_lower': [np.mean(case.cpu_times) - np.std(case.cpu_times) for case in cases_dict.values()],
        'gpu_times_lower': [np.mean(case.gpu_times) - np.std(case.gpu_times) for case in cases_dict.values()],
    }

    source = ColumnDataSource(data=data)

    # Create a new figure
    p = figure(x_range=data['cases_cpu'] + data['cases_gpu'], height=350, title="Performance Comparison",
               toolbar_location=None, tools="")

    # Plot bars for CPU times
    p.vbar(x='cases_cpu', top='cpu_times_mean', width=0.4, source=source, legend_label="CPU times",
           fill_color="#b3de69")

    # Plot bars for GPU times
    p.vbar(x='cases_gpu', top='gpu_times_mean', width=0.4, source=source, legend_label="GPU times",
           fill_color="#7fc97f")

    # Add error bars for CPU times
    p.add_layout(
        Whisker(source=source, base="cases_cpu", upper="cpu_times_upper", lower="cpu_times_lower", level="overlay",
                line_color="#b3de69", line_width=2, line_dash="dashed"))

    # Add error bars for GPU times
    p.add_layout(
        Whisker(source=source, base="cases_gpu", upper="gpu_times_upper", lower="gpu_times_lower", level="overlay",
                line_color="#7fc97f", line_width=2, line_dash="dashed"))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = max(max(data['cpu_times_upper']), max(data['gpu_times_upper'])) * 1.2
    p.legend.location = "top_right"
    p.legend.orientation = "vertical"
    p.legend.label_text_font_size = "10pt"

    # Save to HTML file
    output_file("../py_ml_data/performance_comparison.html")
    save(p)

def plot_whitening_outputs(
    time: np.ndarray, 
    results: dict,
    filename: str
    ) -> None:
    """
    Plot and save the comparison of whitening outputs.

    Parameters
    ----------
    time : numpy.ndarray
        Array of time points.
    whitened_ts : numpy.ndarray
    
        Whitened data from the original function.
    whitened_tf : numpy.ndarray
        Whitened data from the TensorFlow-based function.
    filename : str
        Name of the output HTML file.

    Returns
    -------
    None
    """
    # Create a new figure
    p = figure(title="Comparison of Whitening Outputs", 
               x_axis_label='Time (s)', 
               y_axis_label='Amplitude')
    
    colors = ("orange", "blue", "red", "purple", "green", "indigo", "violet")
    
    dodge = 7
    for i, (color, key) in enumerate(zip(colors, results.keys())):
        p.line(time, results[key] + dodge*i, legend_label=key, line_color=color)

    # Specify the output file and save the plot
    output_file(filename)
    save(p)
    
def plot_whiten_functions(
    sample_rate=8192, 
    duration=16, 
    fftlength=1, 
    overlap=0.5
    ):
    
    cupy.cuda.Device(4).use()
        
    # Generate random data
    np.random.seed(42)    
    t = np.linspace(0, duration, int(duration*sample_rate), endpoint=False)  # time variable
    data = 0.5*np.sin(2*np.pi*1.5*t)+0.1*np.sin(2*np.pi*8.5*t)+0.1*np.random.normal(size=t.shape)
    data = data.astype(np.float32)
    
    f, data_psd = \
        scipy.signal.csd(
            data, 
            data, 
            fs=sample_rate, 
            window='hann', 
            nperseg=int(sample_rate*fftlength), 
            noverlap=int(sample_rate*overlap), 
            average='median')
    
    # Create a GWpy TimeSeries object
    ts = TimeSeries(data, sample_rate=sample_rate)
    
    whitened_ts = ts.whiten(fftlength=fftlength, overlap=overlap).value
    
    f, ts_psd = \
        scipy.signal.csd(
            whitened_ts, 
            whitened_ts, 
            fs=sample_rate, 
            window='hann', 
            nperseg=int(sample_rate*fftlength), 
            noverlap=int(sample_rate*overlap), 
            average='median')

    # Time and run cupy-based function
    timeseries = cupy.asarray(data)
    whitened_cp = whiten(timeseries, timeseries, sample_rate, fftlength=fftlength, overlap=overlap)
    whitened_cp = cupy.asnumpy(whitened_cp)
    
    f, cp_psd = \
        scipy.signal.csd(
            whitened_cp, 
            whitened_cp, 
            fs=sample_rate, 
            window='hann', 
            nperseg=int(sample_rate*fftlength), 
            noverlap=int(sample_rate*overlap), 
            average='median')
    
    residuals = np.abs(whitened_cp - whitened_ts)
    
    results = {"Original": data, "CuPy": whitened_cp, "GWPY": whitened_ts, "Residuals": residuals}
    
    plot_whitening_outputs(t, results, "../py_ml_data/whitening_outputs.html")  
    
    psd_results = {"Original": data_psd, "CuPy": cp_psd, "GWPY": ts_psd}
    plot_whitening_outputs(f, psd_results, "../py_ml_data/psd_outputs.html")  
    
def test_whiten_functions():
    sample_rate = 8192
    duration = 16
    fftlength = 1
    overlap = 0.5

    cupy.cuda.Device(4).use()

    # Generate random data
    np.random.seed(42)
    t = np.linspace(0, duration, int(duration*sample_rate), endpoint=False)
    data = 0.5*np.sin(2*np.pi*1.5*t)+0.1*np.sin(2*np.pi*8.5*t)+0.1*np.random.normal(size=t.shape)
    data = data.astype(np.float32)

    # Create a GWpy TimeSeries object
    ts = TimeSeries(data, sample_rate=sample_rate)
    whitened_ts = ts.whiten(fftlength=fftlength, overlap=overlap).value

    # Whitening using cupy-based function
    timeseries = cupy.asarray(data)
    whitened_cp = whiten(timeseries, timeseries, sample_rate, fftlength=fftlength, overlap=overlap)
    whitened_cp = cupy.asnumpy(whitened_cp)

    # Calculate power spectral densities
    f, data_psd = scipy.signal.csd(data, data, fs=sample_rate, window='hann', nperseg=int(sample_rate*fftlength), noverlap=int(sample_rate*overlap), average='median')
    f, ts_psd = scipy.signal.csd(whitened_ts, whitened_ts, fs=sample_rate, window='hann', nperseg=int(sample_rate*fftlength), noverlap=int(sample_rate*overlap), average='median')
    f, cp_psd = scipy.signal.csd(whitened_cp, whitened_cp, fs=sample_rate, window='hann', nperseg=int(sample_rate*fftlength), noverlap=int(sample_rate*overlap), average='median')
    
    # Find peaks in the PSD of the whitened data
    ts_peaks, _ = scipy.signal.find_peaks(np.abs(ts_psd), threshold = 1.0E-3)
    cp_peaks, _ = scipy.signal.find_peaks(np.abs(cp_psd), threshold = 1.0E-3)
    data_peaks, _ = scipy.signal.find_peaks(np.abs(data_psd), threshold = 1.0E-3)
    
    # Check if there are no peaks in the PSD of the whitened data
    assert len(ts_peaks) == 0, "Peaks found in the PSD of the GWpy whitened data"
    assert len(cp_peaks) == 0, "Peaks found in the PSD of the cupy whitened data"
    
def test_whiten_performace(
    sample_rate=8192, 
    duration=16, 
    fftlength=1, 
    overlap=0.5, 
    num_trials=10
    ):
    
    # Generate random data
    np.random.seed(42)    
    t = np.linspace(0, duration, int(duration*sample_rate), endpoint=False)  # time variable
    data = 0.5*np.sin(2*np.pi*1.5*t)+0.1*np.sin(2*np.pi*8.5*t)+0.1*np.random.normal(size=t.shape)
    data = data.astype(np.float32)
    
    performance_results = {}
    
    performance_results["GWPY"] = benchmark( \
            ts.whiten, 
            (fftlength, overlap), 
            n_repeat=num_trials)
    
    performance_results["CuPy"] = benchmark( \
            whiten, 
            (timeseries, timeseries, sample_rate, fftlength, overlap), 
            n_repeat=num_trials)
    
    assert np.mean(performance_results["CuPy"].cpu_times) < np.mean(performance_results["GWPY"].cpu_times), "GPU version is not faster than CPU version"
    
    plot_performance_comparison(performance_results)
    
test_whiten_functions()
    
def real_noise_test():
    sample_rate_hertz=1024
    fftlength=1 
    overlap=0.5
    num_trials=10
    
    example_duration_seconds = 2.0
    
    background_noise_iterator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = sample_rate_hertz,
        example_duration_seconds = example_duration_seconds,
        max_num_examples = 32,
        num_examples_per_batch = 32,
        order = "shortest_first",
        apply_whitening = False
    )
    
    background_noise_iterator_w = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = sample_rate_hertz,
        example_duration_seconds = example_duration_seconds,
        max_num_examples = 32,
        num_examples_per_batch = 32,
        order = "shortest_first",
        apply_whitening = True
    )
    
    for i, (noise_chunk, noise_chunk_w) in enumerate(zip(background_noise_iterator, background_noise_iterator_w)):
        
        # Convert the TensorFlow tensor to a NumPy array for plotting
        data   = noise_chunk[0].numpy()
        noise_chunk_w = noise_chunk_w[0].numpy()
        
        # Create a GWpy TimeSeries object
        ts = TimeSeries(data, sample_rate=sample_rate_hertz)
        whitened_ts = ts.whiten(fftlength=fftlength, overlap=overlap, fduration = 1.0).value

        # Whitening using cupy-based function
        timeseries = cupy.asarray(noise_chunk.numpy())
        whitened_cp = whiten(timeseries, timeseries, sample_rate_hertz, fftlength=fftlength, overlap=overlap)
        whitened_cp = cupy.asnumpy(whitened_cp[0])

        # Calculate power spectral densities
        f, data_psd = scipy.signal.csd(data, data, fs=sample_rate_hertz, window='hann', nperseg=int(sample_rate_hertz*fftlength), noverlap=int(sample_rate_hertz*overlap), average='median')
        f, ts_psd = scipy.signal.csd(whitened_ts, whitened_ts, fs=sample_rate_hertz, window='hann', nperseg=int(sample_rate_hertz*fftlength), noverlap=int(sample_rate_hertz*overlap), average='median')
        f, cp_psd = scipy.signal.csd(whitened_cp, whitened_cp, fs=sample_rate_hertz, window='hann', nperseg=int(sample_rate_hertz*fftlength), noverlap=int(sample_rate_hertz*overlap), average='median')
        
        residuals = np.abs(whitened_cp - whitened_ts)
            
        results = {"Original": data, "CuPy": whitened_cp, "GWPY": whitened_ts, "Residuals": residuals, "Generator": noise_chunk_w}
        
        t = np.linspace(0, example_duration_seconds, int(sample_rate_hertz*example_duration_seconds))
        plot_whitening_outputs(t, results, "../py_ml_data/rn_whitening_outputs.html")  
    
        psd_results = {"Original": data_psd, "CuPy": cp_psd, "GWPY": ts_psd}
        plot_whitening_outputs(f, psd_results, "../py_ml_data/rn_psd_outputs.html")  

real_noise_test()