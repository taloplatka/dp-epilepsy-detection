FS = 173.61  # Sampling rate of the Bonn EEG Dataset recording in Hz
KMAX = 10  # kmax value for Higuchi FD algorithm
SAMPLES = 4097  # Number of data points in each EEG recording
WINDOW_SIZE = 730  # Window size for slicing EEG time series
OVERLAP = 0.5  # Overlap for EEG time series windows
SEED = 42  # Seed for np.random
TEST_SIZE = 0.3  # Test size for train/test split
K = 5  # K for KNN
