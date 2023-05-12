from whiten import whiten
from bokeh.plotting import figure, save, output_file
from gwpy.timeseries import TimeSeries
import numpy as np
import cupy
import timeit

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
    
    colors = ("orange", "blue", "red")
    
    dodge = 7
    for i, (color, key) in enumerate(zip(colors, results.keys())):
        p.line(time, results[key] + dodge*i, legend_label=key, line_color=color)

    # Specify the output file and save the plot
    output_file(filename)
    save(p)
    
def test_whiten_functions(sample_rate=8192, duration=16, fftlength=4, overlap=2, tol=1e-6):
    
    cupy.cuda.Device(4).use()
        
    # Generate random data
    np.random.seed(42)    
    t = np.linspace(0, duration, int(duration*sample_rate), endpoint=False)  # time variable
    data = 0.5*np.sin(2*np.pi*2*t)+0.1*np.sin(2*np.pi*8*t)+0.1*np.random.normal(size=t.shape)
    data = data.astype(np.float32)
    
    # Create a GWpy TimeSeries object
    ts = TimeSeries(data, sample_rate=sample_rate)
    
    # Time and run original function
    start = timeit.default_timer()
    for i in range(1000):
        whitened_ts = ts.whiten(fftlength=fftlength, overlap=overlap).value
    ts_time = timeit.default_timer() - start

    # Time and run cupy-based function
    timeseries = cupy.asarray(data)
    whitened_cp = whiten(timeseries, sample_rate, fftlength=fftlength, overlap=overlap)
    whitened_cp = cupy.asnumpy(whitened_cp)

    start = timeit.default_timer()
    for i in range(1000):
        whiten(timeseries, sample_rate, fftlength=fftlength, overlap=overlap)
    cp_time = timeit.default_timer() - start
    
    results = {"Original": data, "CuPy": whitened_cp, "GWPY": whitened_ts}
    timings = {"GWPY": ts_time, "CuPy": cp_time}

    plot_whitening_outputs(t, results, "../whitening_outputs.html")
    
    # Print execution times
    for key, val in timings.items():
        print(f"Execution time of {key} function: {val} seconds")

test_whiten_functions()
