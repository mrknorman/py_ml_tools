from .dataset import get_ifo_data_generator, get_ifo_data, O3
from .setup import setup_cuda
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import ColumnDataSource

from itertools import islice
import numpy as np
from tqdm import tqdm

import tensorflow as tf

def test_noise():
            
    background_noise_iterator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = 8*1024.0,
        example_duration_seconds = 1.0,
        num_examples_per_batch = 32,
        order = "random",
        apply_whitening = True
    )
    
    """
    for i, noise_chunk in enumerate(background_noise_iterator):
        
        print(noise_chunk)
        print(i*32)
        
        # Convert the TensorFlow tensor to a NumPy array for plotting
        numpy_array = noise_chunk["data"][0].numpy()

        # Output to an HTML file
        output_file("./py_ml_data/noise_test.html")

        # Create a new plot with a title and axis labels
        p = figure(title="Tensor plot", x_axis_label='x', y_axis_label='y')

        # Add a line renderer with legend and line thickness
        p.line(range(len(numpy_array)), numpy_array, legend_label="Temp.", line_width=2)

        # Show the results
        show(p)
        
        # Convert the TensorFlow tensor to a NumPy array for plotting
        numpy_array = noise_chunk["data"][1].numpy()

        # Output to an HTML file
        output_file("./py_ml_data/noise_test_1.html")

        # Create a new plot with a title and axis labels
        p = figure(title="Tensor plot", x_axis_label='x', y_axis_label='y')

        # Add a line renderer with legend and line thickness
        p.line(range(len(numpy_array)), numpy_array, legend_label="Temp.", line_width=2)

        # Show the results
        show(p)
    """
        
    ifo_data_generator_tf = get_ifo_data_generator(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = 8*1024.0,
        example_duration_seconds = 1.0,
        max_segment_size = 3600,
        num_examples_per_batch = 32,
        order = "random",
        apply_whitening = True,
        return_keys = ["data"]
    ).prefetch(tf.data.AUTOTUNE)
    
    ifo_data_generator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = 8*1024.0,
        example_duration_seconds = 1.0,
        max_segment_size = 3600,
        num_examples_per_batch = 32,
        order = "random",
        apply_whitening = True,
        return_keys = ["data"]
    )
    
    for i in range(2):
        for i, noise_chunk in enumerate(ifo_data_generator_tf):
            print(i*32)
            
            if (i > 32*100):
                break
                
def plot_time_series(onsource, injections, sample_rate_hertz, onsource_duration_seconds, file_name='bokeh_plot.html'):
    # Infer time axis from onsource_duration and sample rate
    time_axis = np.linspace(0, onsource_duration_seconds, onsource.shape[-1])

    # Take the first example from the batches
    onsource_first = onsource[0]
    injections_first = injections[0, 0]

    # Preparing the data
    source = ColumnDataSource(data=dict(time=time_axis, onsource=onsource_first, injections=injections_first))

    # Prepare the output file (HTML)
    output_file(file_name)

    # Create a new plot with a title and axis labels
    p = figure(title="Onsource and Injections over time", x_axis_label='Time (seconds)', y_axis_label='Amplitude')

    # Add a line renderer for 'onsource' (background), line color set to blue
    p.line('time', 'onsource', source=source, line_width=2, line_color="blue", legend_label="Onsource")

    # Add a line renderer for 'injections', line color set to red
    p.line('time', 'injections', source=source, line_width=2, line_color="red", legend_label="Injections")

    # Save the result to HTML file
    save(p)
                
def test_injection(): 
    
    sample_rate_hertz = 8196
    duration_seconds = 2.0
    
    injection_configs = [
        {
            "type" : "cbc",
            "snr"  : 30,
            "injection_chance" : 1.0,
            "padding_seconds" : {"front" : 0.2, "back" : 0.1},
            "args" : {
                "approximant_enum" : \
                    {"value" : 1, "distribution_type": "constant", "dtype" : int}, 
                "mass_1_msun" : \
                    {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                "mass_2_msun" : \
                    {"min_value" : 5, "max_value": 95, "distribution_type": "uniform"},
                "sample_rate_hertz" : \
                    {"value" : sample_rate_hertz, "distribution_type": "constant"},
                "duration_seconds" : \
                    {"value" : duration_seconds, "distribution_type": "constant"},
                "inclination_radians" : \
                    {"min_value" : 0, "max_value": np.pi, "distribution_type": "uniform"},
                "distance_mpc" : \
                    {"min_value" : 10, "max_value": 1000, "distribution_type": "uniform"},
                "reference_orbital_phase_in" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "ascending_node_longitude" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "eccentricity" : \
                    {"min_value" : 0, "max_value": 0.1, "distribution_type": "uniform"},
                "mean_periastron_anomaly" : \
                    {"min_value" : 0, "max_value": 2*np.pi, "distribution_type": "uniform"},
                "spin_1_in" : \
                    {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3},
                "spin_2_in" : \
                    {"min_value" : -0.5, "max_value": 0.5, "distribution_type": "uniform", "num_values" : 3}
            }
        }
    ]
    
    ifo_data_generator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        injection_configs = injection_configs,
        sample_rate_hertz = sample_rate_hertz,
        onsource_duration_seconds = duration_seconds,
        max_segment_size = 3600,
        num_examples_per_batch = 32,
        order = "random",
        apply_whitening = True,
        return_keys = ["onsource", "injections"],
        save_segment_data = True,
    )
    
    for data in islice(ifo_data_generator, 1):
        
        plot_time_series(
            data['onsource'].numpy(), 
            data['injections'].numpy(), 
            sample_rate_hertz, 
            duration_seconds, 
            file_name='./py_ml_data/injection_test.html'
        )
        
    for data in tqdm(islice(ifo_data_generator, 100)):
        pass

                
if __name__ == "__main__":
    setup_cuda("0", verbose = True)    
    test_injection()
    #test_noise()

    
   