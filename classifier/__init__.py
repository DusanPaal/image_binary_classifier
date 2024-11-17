# standard library imports
import os

# NOTE: Tensorflow log level must be set before the tensorflow is imported. 
KERAS_BACKEND_TF = 'tensorflow'
KERAS_BACKEND_TA = 'theano'
KERAS_BACKEND_PT = 'pytorch'

os.environ['KERAS_BACKEND'] = KERAS_BACKEND_TF

# NOTE: Keras should only be imported after the backend has been configured.
# The backend cannot be changed once the package is imported.
TF_LOG_ALL = '0'        # all messages (default)
TF_LOG_INFO = '1'       # info logs
TF_LOG_WARNING = '2'    # warning messages
TF_LOG_ERROR = '3'      # error messages

tf_log_level = TF_LOG_INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level


# now that keras backend and tensorflow
# log level are set, we can import tensorflow
import tensorflow as tf
import tensorflow.config.experimental as tf_ecfg

class System:
    '''Provides utilities for
    configuring the system.'''

    # NOTE: imlementing GPU and CPU detection 
    # as a separate function in case we decide 
    # # to amke it public at a later stage
    @staticmethod
    def _detect_devices(dev: str) -> list:
        '''Detects the CPU and GPU devices
        available to TensorFlow.

        Parameters:
        -----------
        dev : str
            The device to detect:
            - 'CPU': Central Processing Unit
            - 'GPU: Graphics Processing Unit

        Returns:
        --------
        list
            A list of devices available to TensorFlow.
        '''

        if dev.upper() not in ['CPU', 'GPU']:
            raise ValueError(
                "Invalid device! "
                "Choose from 'CPU' or 'GPU'."
            )
        
        return tf.config.list_physical_devices(dev.upper())

    @staticmethod
    def print_info():
        '''Prints information about the setup of the system.'''

        print("System Information:")
        print('-------------------')

        keras_backend = os.environ.get('KERAS_BACKEND', None)
        print(f"Keras Backend: {keras_backend}")

        tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', TF_LOG_ALL)
        print(f"Tensorflow log level: {tf_log_level}")

        # List available physical devices
        print("CPUs Available:", System._detect_devices('CPU'))
        print("GPUs Available:", System._detect_devices('GPU'))

        print('-------------------')

    @staticmethod
    def configure_gpu_allocation(gpu=None, limit=None, dynamic=False) -> None:
        '''Enables incremental GPU memory allocation.

        Memory growth must be set before GPU initialization.
        
        Parameters:
        -----------
        gpu : int
            The GPU device to set the memory growth on.
            If None, the memory growth is set on all GPUs.

        limit : int
            The memory limit (in MB)to set on the GPU device.
        '''

        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            raise ValueError(
                'Invalid memory limit! Memory limit '
                'must be a positive integer.'
            )

        if gpu is not None:
            gpus = [gpu]
        else:
            gpus = tf_ecfg.list_physical_devices('GPU')

        if len(gpus) == 0:
            print('Error: Cannot configure memory allocation. No GPU devices found!')
            return
        
        for gpu in gpus:
            if dynamic:
                try:
                    tf_ecfg.set_memory_growth(gpu, True)
                except RuntimeError as err:
                    print(err)
            if limit is not None:
                try:
                    tf_ecfg.set_virtual_device_configuration(
                        gpu,[tf_ecfg.VirtualDeviceConfiguration(memory_limit=limit)]
                    )
                    print(f'Tensorflow will use {limit} MB of all GPU memory.')
                except RuntimeError as err:
                    print(err)
        