from .dataset import get_ifo_data, O3
from .cuphenom.py.cuphenom import generate_phenom
from .whiten import whiten
from .snr import scale_to_snr
import tensorflow as tf
import numpy as np

from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Category10

def plot_and_save(background, injection, o_injection, scaled_injection, signal_duration):
    # convert tensors to numpy arrays
    background = background.numpy()
    injection = injection.numpy()
    scaled_injection = scaled_injection.numpy()
    o_injection = o_injection.numpy()

    # Prepare output file
    output_file("timeseries_plot.html")

    # Create a new plot with a title and axis labels
    p1 = figure(title="Background", x_axis_label='x', y_axis_label='y')
    p2 = figure(title="Injection", x_axis_label='x', y_axis_label='y')
    p3 = figure(title="Scaled Injection", x_axis_label='x', y_axis_label='y')

    # Add a line renderer with legend and line thickness
    
    x = np.linspace(0, signal_duration, len(background))
    p = figure(width=600, height=400)

    # Here x-values are assumed to be the indices. 
    # If you have a separate x-array, use that instead.

    # Plot background
    p.line(x, background, legend_label="Background", line_color=Category10[3][0])

    # Plot injection
    p.line(x, injection, legend_label="Whitened Injection", line_color=Category10[3][1])

    # Plot scaled_injection
    p.line(x, scaled_injection, legend_label="Whitened Scaled Injection", line_color=Category10[3][2])
    
    p.line(x, o_injection, legend_label="Original Injection", line_color=Category10[4][3])


    p.legend.location = "top_left"

    # Save the html
    save(p)

if __name__ == "__main__":
    # Call generatePhenom function
    
    sample_rate_hertz = 4096
    example_duration_seconds = 8.0
    
    background_noise_iterator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = sample_rate_hertz,
        example_duration_seconds = example_duration_seconds,
        num_examples_per_batch = 32,
        order = "shortest_first",
        apply_whitening = False,
        return_keys = ["data", "background"]
    )
    
    for background in background_noise_iterator:
        
        injection = \
            generate_phenom(
                approximant_enum = 1, 
                mass_1_msun = 30,
                mass_2_msun = 30,
                sample_rate_hertz = sample_rate_hertz,
                duration_seconds = example_duration_seconds,
                inclination_radians = 1.0,
                distance_mpc = 100,
                reference_orbital_phase_in = 0.0,
                ascending_node_longitude = 100.0,
                eccentricity = 0.0,
                mean_periastron_anomaly = 0.0, 
                spin_1_in = [0.0, 0.0, 0.0],
                spin_2_in = [0.0, 0.0, 0.0]
            )
        
        injection *= 1.0E20

        #some kind of projection here
        
        injection = tf.convert_to_tensor(injection[:, 1], dtype = tf.float32)
        injection = tf.roll(injection, len(injection)//2, -1)
        real_background = tf.cast(background["background"][0], dtype = tf.float32)
        background = tf.cast(background["data"][0], dtype = tf.float32)
        
        scaled_injection = \
            scale_to_snr(
                injection, 
                background,
                30.0,
                window_duration_seconds = 2.0, 
                sample_rate_hertz = sample_rate_hertz, 
                fft_duration_seconds = 2.0, 
                overlap_duration_seconds = 1.0,
            )        
        
        background = background + injection

        
        background = whiten(
            background, 
            real_background, 
            sample_rate_hertz, 
            fftlength=2.0, 
            overlap=1.0
        )
        
        o_injection = scaled_injection
        
        injection = whiten(
            injection, 
            real_background, 
            sample_rate_hertz, 
            fftlength=2.0, 
            overlap=1.0
        )
        
        scaled_injection = whiten(
            scaled_injection, 
            real_background, 
            sample_rate_hertz, 
            fftlength=2.0, 
            overlap=1.0
        )
        
        plot_and_save(background, injection, o_injection, scaled_injection, example_duration_seconds)
                        
        quit()
    
