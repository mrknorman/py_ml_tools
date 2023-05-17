from dataclasses import dataclass
from typing import List, Tuple, Union
from datetime import datetime
import gc

import numpy as np
import hashlib
import tensorflow as tf
from gwdatafind import find_urls
from gwpy.segments import DataQualityDict
from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import cupy
from .whiten import whiten
from .setup import cupy_to_tensor
import cusignal
from cusignal.filtering.resample import decimate
from cupyx.profiler import benchmark

import h5py

@dataclass
class ObservingRun:
    name: str
    start_date_time: datetime
    end_date_time: datetime
    channels: dict
    frame_types: dict
    state_flags: dict

    def __post_init__(self):
        self.start_gps_time = self._to_gps_time(self.start_date_time)
        self.end_gps_time = self._to_gps_time(self.end_date_time)

    @staticmethod
    def _to_gps_time(date_time: datetime) -> float:
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        time_diff = date_time - gps_epoch
        leap_seconds = 18  # current number of leap seconds as of 2021 (change if needed)
        total_seconds = time_diff.total_seconds() - leap_seconds
        return total_seconds


IFOS = ("H1", "L1", "V1")

observing_run_data = (
    ("O1", datetime(2015, 9, 12, 0, 0, 0), datetime(2016, 1, 19, 0, 0, 0),
     {"best": "DCS-CALIB_STRAIN_CLEAN_C01"},
     {"best": "HOFT_C01"},
     {"best": "DCS-ANALYSIS_READY_C01:1"}),
    ("O2", datetime(2016, 11, 30, 0, 0, 0), datetime(2017, 8, 25, 0, 0, 0),
     {"best": "DCS-CALIB_STRAIN_CLEAN_C01"},
     {"best": "HOFT_C01"},
     {"best": "DCS-ANALYSIS_READY_C01:1"}),
    ("O3", datetime(2019, 4, 1, 0, 0, 0), datetime(2020, 3, 27, 0, 0, 0),
     {"best": "DCS-CALIB_STRAIN_CLEAN_C01"},
     {"best": "HOFT_C01"},
     {"best": "DCS-ANALYSIS_READY_C01:1"})
)

OBSERVING_RUNS = {name: ObservingRun(name, start_date_time, end_date_time, channels, frame_types, state_flags)
                  for name, start_date_time, end_date_time, channels, frame_types, state_flags in observing_run_data}

O1 = OBSERVING_RUNS["O1"]
O2 = OBSERVING_RUNS["O2"]
O3 = OBSERVING_RUNS["O3"]  

@dataclass
class IFOData:
    data        : Union[TimeSeries, cupy.ndarray, np.ndarray]
    t0          : float
    sample_rate : float
    
    def __post_init__(self):
        if (type(self.data) == TimeSeries):
            self.data = cupy.asarray(self.data.value, dtype=cupy.float32)
        elif (type(self.data) == np.ndarray):
            self.data = cupy.asarray(self.data, dtype=cupy.float32)
        
        self.duration = len(self.data)/self.sample_rate
        self.dt = 1.0/self.sample_rate
            
    def downsample(self, new_sample_rate: Union[int, float]):        
        factor = int(self.sample_rate / new_sample_rate)

        new_data = decimate(self.data, factor)
        del self.data  # release the memory     
        self.data = new_data
        gc.collect()
        
        self.sample_rate = new_sample_rate
        
        return self
    
    def scale(self, scale_factor:  Union[int, float]):
        self.data *= scale_factor
        return self
    
    def numpy(self):
        """Converts the data to a numpy array."""
        return cupy.asnumpy(self.data)
    
    def random_subsection(self, num_subsection_elements: int, num_background_elements: int, num_examples_per_batch: int):      
        # Check if the input array is 1D
        assert len(self.data.shape) == 1, "Input array must be 1D"

        # Get the length of the input array
        N = self.data.shape[0]

        # Ensure num_subsection_elements + num_background_elements is smaller or equal to N
        assert num_subsection_elements + num_background_elements <= N, "num_subsection_elements + num_background_elements must be smaller or equal to the length of the array"

        # Generate a random starting index for each element in the batch
        maxval = N - num_subsection_elements - num_background_elements + 1
        random_starts = cupy.random.randint(num_background_elements, maxval, size=(num_examples_per_batch,))

        # Extract the subsections of the array for the entire batch
        indices = cupy.expand_dims(cupy.arange(num_subsection_elements), 0) + random_starts[:, cupy.newaxis]
        batch_subarrays = self.data[indices]

        # Extract the background chunks of the array for the entire batch
        bg_indices = cupy.expand_dims(cupy.arange(num_background_elements), 0) + (random_starts - num_background_elements)[:, cupy.newaxis]
        batch_background_chunks = self.data[bg_indices]

        # Calculate the t0 for each captured subsection
        t0_subsections = self.t0 + cupy.asnumpy(random_starts).astype(cupy.float32) * self.dt

        return batch_subarrays, batch_background_chunks, t0_subsections
    
def get_segment_times(
    start: float,
    stop: float,
    ifo: str,
    state_flag: str
    ) -> list:
    
    segments = DataQualityDict.query_dqsegdb(
        [f"{ifo}:{state_flag}"],
        start,
        stop,
    )
    
    intersection = segments[f"{ifo}:{state_flag}"].active.copy()
    
    return np.array(intersection)
    
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
    ) -> np.ndarray:
    left = arr - offset
    right = arr + increment
    result = np.stack((left, right), axis=1)
    return result

def compress_periods(periods: np.ndarray) -> np.ndarray:
    periods = periods[periods[:,0].argsort()]
    compressed = []

    for period in periods:
        if not compressed or compressed[-1][1] < period[0]:
            compressed.append(period)
        else:
            compressed[-1] = (compressed[-1][0], max(compressed[-1][1], period[1]))

    return np.array(compressed)

def remove_overlap(start: float, end: float, veto_periods: np.ndarray) -> np.ndarray:
    result = np.array([[start, end]])
    for veto_start, veto_end in veto_periods:
        new_result = []
        for period_start, period_end in result:
            if period_start < veto_start < period_end and period_start < veto_end < period_end:
                new_result.append([period_start, veto_start])
                new_result.append([veto_end, period_end])
            elif veto_start <= period_start < veto_end < period_end:
                new_result.append([veto_end, period_end])
            elif period_start < veto_start < period_end <= veto_end:
                new_result.append([period_start, veto_start])
            elif veto_end <= period_start or period_end <= veto_start:
                new_result.append([period_start, period_end])
        result = np.array(new_result)
    return result

def veto_time_periods(valid_periods: np.ndarray, veto_periods: np.ndarray) -> np.ndarray:
    valid_periods = compress_periods(valid_periods)
    veto_periods = compress_periods(veto_periods)
    result = np.vstack([remove_overlap(valid_start, valid_end, veto_periods) for valid_start, valid_end in valid_periods])
    return result

def split_periods(periods: np.ndarray, max_length: float) -> np.ndarray:
    result = []
    for start, end in periods:
        n_splits = int(np.ceil((end - start) / max_length))
        starts = np.linspace(start, start + max_length * (n_splits - 1), n_splits)
        ends = np.minimum(starts + max_length, end)
        result.append(np.vstack((starts, ends)).T)
    return np.vstack(result)

def remove_short_periods(periods: np.ndarray, min_length: float) -> np.ndarray:
    return periods[np.where(periods[:, 1] - periods[:, 0] >= min_length)]

def open_hdf5_file(
    file_path : Union[str, Path], 
    mode : str ='r+'
    ) -> h5py.File:
    
    file_path = Path(file_path)
    try:
        # Try to open the HDF5 file in the specified mode
        f = h5py.File(file_path, mode)
        f.close()
    except OSError:
        # The file does not exist, so create it in write mode
        f = h5py.File(file_path, 'w')
        f.close()
        print(f'The file {file_path} was created in write mode.')
    else:
        print(f'The file {file_path} was opened in {mode} mode.')
    return h5py.File(file_path, mode)

def ensure_directory_exists(
    directory: Union[str, Path]
    ):
    
    directory = Path(directory)  # Convert to Path if not already
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def get_ifo_data(
    time_interval: Union[tuple, ObservingRun], 
    data_labels: List[str], 
    ifo: str,
    sample_rate_hertz: float,    
    data_quality: str = "best",
    channel: str = None,
    frame_type: str = None,
    state_flag: str = None,
    saturation: float = 1.0,
    example_duration_seconds: float = 1.0,
    background_duration_seconds: float = 16.0,
    max_num_examples: float = 0.0,
    apply_whitening: bool = False,
    num_examples_per_batch: int = 1,
    scale_factor: float = 1.0e20,
    max_segment_size = 2000,
    order: str = "random",
    seed: int = 1000,
    force_generation: bool = False,
    data_directory: Union[str, Path] = "../generator_data",
    save_segment_data: bool = False,
    return_keys = ["data", "background", "gps_time"],
    fduration = 1.0
):
    data_directory = Path(data_directory)
    ensure_directory_exists(data_directory)
    
    cupy.random.seed(seed=seed)
    np.random.seed(seed)
    
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

        return data

    if isinstance(time_interval, tuple):
        start, stop = time_interval
    elif isinstance(time_interval, ObservingRun):
        start, stop = time_interval.start_gps_time, time_interval.end_gps_time    

        if (frame_type == None):
            frame_type = time_interval.frame_types[data_quality]
        if (channel == None):
            channel = time_interval.channels[data_quality]
        if (state_flag == None):
            state_flag = time_interval.state_flags[data_quality]
    else:
        raise TypeError("time_interval must be either a tuple or a ObservingRun object")
    
    # Generate a hash from the input parameters to use as a unique identifier
    segment_parameters = [frame_type, channel, state_flag, str(data_labels), sample_rate_hertz] # get a dictionary of all parameters
    param_string = str(segment_parameters)  # convert the dictionary to a string
    param_hash = hashlib.sha1(param_string.encode()).hexdigest()
    segment_filename = data_directory / f"segment_data_{param_hash}.hdf5"
    
    valid_segments = get_segment_times(
        start,
        stop,
        ifo,
        state_flag
    )
    
    veto_segments = []
    if "events" not in data_labels:
        event_times = get_all_event_times()
        veto_segments.append(pad_gps_times_with_veto_window(event_times))
    if "glitches" not in data_labels:
        #glitches = EventTable.fetch('gravityspy', 'glitches')
        #print(glitches)
        print("Glitch vetos not implemented!")
        pass
    
    veto_segments = np.concatenate(veto_segments)

    valid_segments = veto_time_periods(valid_segments, veto_segments)
    valid_segments = split_periods(valid_segments, max_segment_size)
    valid_segments = remove_short_periods(
        valid_segments, 
        (example_duration_seconds + fduration)*num_examples_per_batch 
        + background_duration_seconds
    )
    
    if (len(valid_segments) == 0) and (verbosity >= 1):
        raise ValueError("No valid segments!")

    if order == "random":
        np.random.shuffle(valid_segments)
    elif order == "shortest_first":
        sort_by_duration = lambda segments: segments[np.argsort(segments[:, 1] - segments[:, 0])]
        valid_segments = sort_by_duration(valid_segments)
    elif order == "chronological":
        pass

    current_example_index = 0

    with open_hdf5_file(segment_filename) as f:
        
        f.require_group("segments")
        for current_segment_start, current_segment_end in valid_segments:
            
            current_max_batch_count = int((current_segment_end - current_segment_start) / (saturation * num_examples_per_batch))
            segment_key = f"segments/segment_{current_segment_start}_{current_segment_end}"
                            
            current_segment_data = None

            if (segment_key in f) and save_segment_data:
                print(f"Reading segments of duration {current_segment_end - current_segment_start}...")
                current_segment_data = IFOData(f[segment_key][()], current_segment_start, sample_rate_hertz)
            else: 
                print(f"Acquiring segments of duration {current_segment_end - current_segment_start}...")
                try:
                    current_segment_data = get_new_segment_data(current_segment_start, current_segment_end)
                except Exception as e:
                    print(f"Unexpected error: {type(e).__name__}, {str(e)}")
                    continue
                
                current_segment_data = \
                    IFOData(
                        current_segment_data, 
                        current_segment_data.t0.value, 
                        current_segment_data.sample_rate.value
                    )
                current_segment_data = current_segment_data.downsample(sample_rate_hertz) 
                
                if save_segment_data:
                    f.create_dataset(segment_key, data = current_segment_data.numpy())
            print("Complete!")

            current_segment_data = current_segment_data.scale(scale_factor)                

            for _ in range(current_max_batch_count):
                num_subsection_elements = int((example_duration_seconds + fduration) * sample_rate_hertz)
                num_background_elements = int(background_duration_seconds * sample_rate_hertz)
                batched_examples, batched_backgrounds, batched_gps_times = current_segment_data.random_subsection(num_subsection_elements, num_background_elements, num_examples_per_batch)
                                
                # Injection, projection
                if apply_whitening:
                    #cusignal.spectral_analysis.spectral.csd(batched_examples[1], batched_examples[1])
                    batched_examples = whiten(
                        batched_examples, 
                        batched_backgrounds, 
                        sample_rate_hertz, 
                        fftlength = 1.0,
                        overlap = 0.5,
                        fduration = fduration)
                    
                # Crop to remove edge effects, crop with or without whitening to
                # ensure same data is retrieve in both cases
                desired_num_samples = int(example_duration_seconds * sample_rate_hertz)
                start = (batched_examples.shape[-1] - desired_num_samples) // 2
                end = start + desired_num_samples
                batched_examples = batched_examples[:, start:end]
                batched_examples = cupy.ascontiguousarray(batched_examples)
                
                current_example_index += num_examples_per_batch;
                
                if (max_num_examples > 0) and (current_example_index > max_num_examples):
                    return
                
                return_dict = {}
                if 'data' in return_keys:
                    return_dict['data'] = tf.cast(cupy_to_tensor(batched_examples), tf.float16)
                if 'background' in return_keys:
                    return_dict['background'] = tf.cast(cupy_to_tensor(batched_backgrounds), tf.float16)
                if 'gps_time' in return_keys:
                    return_dict['gps_time'] = tf.convert_to_tensor(batched_gps_times, dtype=tf.int64)
                
                yield return_dict

def get_ifo_data_generator(
    time_interval: Union[tuple, ObservingRun], 
    data_labels: List[str], 
    ifo: str,
    sample_rate_hertz: float,    
    return_keys = ["data", "background", "gps_time"],
    **kwargs  # Capture all other arguments
    ):
    
    output_signature = {
        'data'      : tf.TensorSpec(shape=(kwargs.get('num_examples_per_batch', 1), int(kwargs.get('example_duration_seconds', 1.0)*sample_rate_hertz)), dtype=tf.float16),
        'background': tf.TensorSpec(shape=(kwargs.get('num_examples_per_batch', 1), int(kwargs.get('background_duration_seconds', 16.0)*sample_rate_hertz)), dtype=tf.float16),
        'gps_time'  : tf.TensorSpec(shape=(kwargs.get('num_examples_per_batch', 1)), dtype=tf.int64),
    }
    
    output_signature = {k: output_signature[k] for k in return_keys}
    
    generator = lambda: get_ifo_data(time_interval, data_labels, ifo, sample_rate_hertz, return_keys = return_keys, **kwargs)
    
    return tf.data.Dataset.from_generator(
            generator = generator,
            output_signature = output_signature
        )