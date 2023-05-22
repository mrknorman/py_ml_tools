from .dataset import get_ifo_data_generator, get_ifo_data, O3
from .setup import setup_cuda
from bokeh.plotting import figure, output_file, show

import cupy as cp
import tensorflow as tf

def test_noise():
            
    background_noise_iterator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = 8*1024.0,
        example_duration_seconds = 1.0,
        max_num_examples = 32,
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
        max_num_examples = 1e4,
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
        max_num_examples = 1e4,
        num_examples_per_batch = 32,
        order = "random",
        apply_whitening = True,
        return_keys = ["data"]
    )
    
    for i in range(2):
        for i, noise_chunk in enumerate(ifo_data_generator_tf):
            print(i*32)
        
        
if __name__ == "__main__":
    setup_cuda("0", verbose = True)    

    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=4*1024**3)  # 1 GiB

    test_noise()

    
   