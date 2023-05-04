from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable

@dataclass
class NoiseChunk:
    noise : tf.Tensor
    t0    : float
    dt    : float
    
    def random_subsection(self, num_subsection_elements: int, num_examples_per_batch: int):      
        # Check if the input tensor is 1D
        assert len(self.noise.shape) == 1, "Input tensor must be 1D"

        # Get the length of the input tensor
        N = self.noise.shape[0]

        # Ensure X is smaller or equal to N
        assert num_subsection_elements <= N, "X must be smaller or equal to the length of the tensor"

        # Generate a random starting index for each element in the batch
        maxval = tf.cast(N - num_subsection_elements + 1, dtype=tf.int32)
        random_starts = tf.random.uniform((num_examples_per_batch,), minval=0, maxval=maxval, dtype=tf.int32)

        # Extract the subsections of the tensor for the entire batch
        indices = tf.expand_dims(tf.range(num_subsection_elements, dtype=tf.int32), 0) + random_starts[:, tf.newaxis]
        batch_subtensors = tf.gather(self.noise, indices)

        # Calculate the t0 for each captured subsection
        t0_subsections = self.t0 + tf.cast(random_starts, dtype=tf.float32) * self.dt

        return tf.expand_dims(batch_subtensors, axis=-1), t0_subsections
    
def timeseries_to_noise_chunk(timeseries : TimeSeries, scale_factor : float):
    data = tf.convert_to_tensor(timeseries.value * scale_factor, dtype = tf.float32)
    return NoiseChunk(data, timeseries.t0.value, timeseries.dt.value)

def get_segment_times(
    start: float,
    stop: float,
    ifo: str,
    state_flag: str,
    minimum_duration: float,
    verbosity: int
    ) -> list:
    
    segments = DataQualityDict.query_dqsegdb(
        [f"{ifo}:{state_flag}"],
        start,
        stop,
    )

    intersection = segments[f"{ifo}:{state_flag}"].active.copy()
    
    valid_segments = []
    for seg_start, seg_stop in intersection:
        if (seg_stop - seg_start) >= minimum_duration:
            valid_segments.append((seg_start, seg_stop))

    if not valid_segments and (verbosity >= 1):
        raise ValueError("No segments of minimum length.")
    
    return valid_segments

def get_all_event_times() -> np.ndarray:
    
    catalogues = [
        "GWTC", 
        "GWTC-1-confident", 
        "GWTC-1-marginal", 
        "GWTC-2", 
        "GWTC-2.1-auxiliary", 
        "GWTC-2.1-confident", 
        "GWTC-2.1-marginal", 
        "GWTC-3-confident", 
        "GWTC-3-marginal"
    ]
    
    gps_times = np.array([])
    for catalogue in catalogues:
        events = EventTable.fetch_open_data(catalogue)
        gps_times = np.append(gps_times, events["GPS"].data.compressed())
        
    return gps_times    

def pad_gps_times_with_veto_window(
        arr: np.ndarray, 
        offset: int = 60, 
        increment: int = 10
    ) -> list:
    left = arr - offset
    right = arr + increment
    result = np.stack((left, right), axis=1)
    tuple_result = [tuple(pair) for pair in result]
    return tuple_result

def compress_periods(periods: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    sorted_periods = sorted(periods, key=lambda x: x[0])
    compressed = []

    for period in sorted_periods:
        if not compressed or compressed[-1][1] < period[0]:
            compressed.append(period)
        else:
            compressed[-1] = (compressed[-1][0], max(compressed[-1][1], period[1]))

    return compressed

def remove_overlap(start: int, end: int, veto_periods: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    result = [(start, end)]
    for veto_start, veto_end in veto_periods:
        new_result = []
        for period_start, period_end in result:
            if period_start < veto_start < period_end and period_start < veto_end < period_end:
                new_result.append((period_start, veto_start))
                new_result.append((veto_end, period_end))
            elif veto_start <= period_start < veto_end < period_end:
                new_result.append((veto_end, period_end))
            elif period_start < veto_start < period_end <= veto_end:
                new_result.append((period_start, veto_start))
            elif not (veto_end <= period_start or period_end <= veto_start):
                continue
            else:
                new_result.append((period_start, period_end))
        result = new_result
    return result

def veto_time_periods(valid_periods: List[Tuple[int, int]], veto_periods: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    valid_periods = compress_periods(valid_periods)
    veto_periods = compress_periods(veto_periods)
    
    result = [time_period for valid_start, valid_end in valid_periods for time_period in remove_overlap(valid_start, valid_end, veto_periods) if time_period]
    return result
    
def get_ifo_data(
    start: float,
    stop: float,
    data_labels: List[str], 
    ifo: str,
    sample_rate_hertz: float,
    channel: str,
    frame_type: str,
    state_flag: str,
    saturation: float = 1.0,
    example_duration_seconds: float = 1.0,
    max_num_examples: float = 0.0,
    num_examples_per_batch: int = 1,
    scale_factor: float = 1.0,
    order: str = "random",
):
    valid_segments = get_segment_times(
        start,
        stop,
        ifo,
        state_flag,
        example_duration_seconds*num_examples_per_batch,
        0
    )
    
    veto_segments = []
    if "events" not in data_labels:
        event_times = get_all_event_times()
        veto_segments += pad_gps_times_with_veto_window(event_times)
    if "glitches" not in data_labels:
        #glitches = EventTable.fetch('gravityspy', 'glitches')
        #print(glitches)
        print("Glitch vetos not implemented!")
        pass
    
    valid_segments = veto_time_periods(valid_segments, veto_segments)
    
    if order == "random":
        shuffle(valid_segments)
    elif order == "shortest_first":
        valid_segments = sorted(valid_segments, key=lambda duration: duration[1] - duration[0])

    def get_new_segment_data(segment_start, segment_end):
        files = find_urls(
            site=ifo.strip("1"),
            frametype=f"{ifo}_{frame_type}",
            gpsstart=segment_start,
            gpsend=segment_end,
            urltype="file",
        )

        data = TimeSeries.read(
            files, channel=f"{ifo}:{channel}", start=segment_start, end=segment_end, nproc=4
        )
        data = data.resample(sample_rate_hertz)
        data = data.whiten(4, 2)

        if np.isnan(data).any():
            raise ValueError(f"The noise for ifo {ifo} contains NaN values")

        return timeseries_to_noise_chunk(data, scale_factor)

    current_segment_index = 0
    current_example_index = 0

    while current_segment_index < len(valid_segments):
        current_segment_start, current_segment_end = valid_segments[current_segment_index]
        current_max_sample_count = int((current_segment_end - current_segment_start) / saturation)

        try:
            print(f"Loading segments of duration {current_segment_end - current_segment_start}")
            current_segment_data = get_new_segment_data(current_segment_start, current_segment_end)
            print("Complete!")

            for _ in range(current_max_sample_count):
                num_subsection_elements = int(example_duration_seconds * sample_rate_hertz)
                batch_noise_data = current_segment_data.random_subsection(num_subsection_elements, num_examples_per_batch)
                
                current_example_index += num_examples_per_batch;
                
                if (max_num_examples > 0) and (current_example_index > max_num_examples):
                    print("Exausted Examples! Exiting!")
                    return
                                    
                yield batch_noise_data

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}, {str(e)}")

        current_segment_index += 1
