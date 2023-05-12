import tensorflow as tf
import os

def setup_cuda(
    device_num : str,
    verbose : bool = False
    ):
		
	os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
		
	gpus =  tf.config.list_logical_devices('GPU')
	strategy = tf.distribute.MirroredStrategy(gpus)

	physical_devices = tf.config.list_physical_devices('GPU')
	
	for device in physical_devices:	

		try:
			tf.config.experimental.set_memory_growth(device, True)
		except:
			# Invalid device or cannot modify virtual devices once initialized.
			pass
	
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	if verbose:
		tf.config.list_physical_devices("GPU")
		
	return strategy