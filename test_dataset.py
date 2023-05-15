from dataset import get_ifo_data, O3
from setup import setup_cuda

def test_noise():
            
    background_noise_iterator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        ifo = "L1",
        sample_rate_hertz = 1024.0,
        example_duration_seconds = 1.0,
        max_num_examples = 32,
        num_examples_per_batch = 32,
        order = "shortest_first",
        apply_whitening = True
    )
    
    for i, noise_chunk in enumerate(background_noise_iterator):
        
        print(noise_chunk)
        print(i*32)
        
if __name__ == "__main__":
    setup_cuda("5")
    test_noise()

    
   