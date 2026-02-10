import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import numpy as np
import scipy.signal
import tensorflow as tf
import librosa
from tensorflow.image import resize

# ==========================================
# 1. CONFIGURATION (Matched to your Max_Final Code)
# ==========================================
MODEL_PATH = "Trained_model_REPET_Spectrogram.h5"
CLASSES = ['Healthy', 'Off', 'Belt_Fault']

# Hardware Params
ADS_DATA_RATE = 860     # Matches your 'ADS_DATA_RATE'
GAIN = 16               # Matches your 'GAIN'
CHANNEL = 0             # Matches your 'CHANNEL'

# Prediction Params
CHUNK_DURATION = 4      # Seconds
BUFFER_SIZE = int(ADS_DATA_RATE * CHUNK_DURATION) # 3440 samples

# ==========================================
# 2. HARDWARE SETUP
# ==========================================
# Using standard I2C (Speed controlled by OS config, just like your recorder)
i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)
ads.data_rate = ADS_DATA_RATE
ads.gain = GAIN
chan = AnalogIn(ads, CHANNEL)

print(f"=== LIVE FAULT DETECTOR ===")
print(f"Sensor: ADS1115 | Rate: {ADS_DATA_RATE} SPS | Gain: {GAIN}")
print(f"Buffer: {BUFFER_SIZE} samples per prediction")
print("Loading Model... (This may take 10-20 seconds)")

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)
print("✓ Model Loaded. Starting Loop...")
print("-" * 40)

# ==========================================
# 3. REPET FUNCTION (Must Match Training)
# ==========================================
def get_repet_foreground_spec(audio_signal, sampling_frequency):
    """
    Extracts foreground spectrogram using REPET.
    """
    # A. Setup STFT
    window_length = int(pow(2, np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.windows.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # B. Compute STFT
    S = librosa.stft(audio_signal, n_fft=window_length, hop_length=step_length, window=window_function)
    S_mag = np.abs(S)

    # C. Estimate Period
    beat_spectrum = np.mean(S_mag ** 2, axis=0)
    beat_spectrum = beat_spectrum - np.mean(beat_spectrum)
    autocorrelation = np.correlate(beat_spectrum, beat_spectrum, mode='full')
    autocorrelation = autocorrelation[len(autocorrelation) // 2:]
    
    min_lag = int(0.1 * sampling_frequency / step_length)
    max_lag = len(autocorrelation) // 2
    
    if min_lag >= max_lag:
        period = 1
    else:
        period = np.argmax(autocorrelation[min_lag:max_lag]) + min_lag

    # D. Background Model
    n_freq, n_time = S_mag.shape
    num_segments = int(np.ceil(n_time / period))
    pad_length = num_segments * period - n_time
    
    S_mag_padded = np.pad(S_mag, ((0, 0), (0, pad_length)), 'constant')
    S_reshaped = S_mag_padded.reshape(n_freq, period, num_segments, order='F')
    S_repeating = np.median(S_reshaped, axis=2)
    S_background_padded = np.tile(S_repeating, (1, num_segments))
    S_background_mag = S_background_padded[:, :n_time]

    # E. Mask & Foreground
    eps = np.finfo(float).eps
    mask = S_background_mag / (S_mag + eps)
    mask = np.minimum(mask, 1.0)
    S_foreground_mag = S_mag - (mask * S_mag)
    S_foreground_mag = np.maximum(S_foreground_mag, 0)

    return S_foreground_mag

# ==========================================
# 4. PREPROCESSING (Raw -> AI Input)
# ==========================================
def preprocess_live_chunk(raw_samples):
    # 1. Convert List to Numpy Array
    chunk = np.array(raw_samples, dtype=np.float32)
    
    # 2. REMOVE DC OFFSET
    # Your recording code saves raw data (approx 1.65V).
    # The AI was trained on centered data (0V). We fix that here.
    chunk = chunk - np.mean(chunk)

    # 3. Apply REPET & Spectrogram
    try:
        # Get Foreground Spectrogram
        foreground_spec = get_repet_foreground_spec(chunk, ADS_DATA_RATE)
        
        # Convert to Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            S=foreground_spec,
            sr=ADS_DATA_RATE,
            n_mels=128
        )
    except Exception:
        # Fallback if REPET fails (e.g., silence)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk, sr=ADS_DATA_RATE, n_mels=128
        )

    # 4. Convert to dB and Resize
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Resize to (128, 54) - The input shape of your CNN
    target_shape = (128, 54)
    resized_spec = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    
    # Add Batch Dimension -> (1, 128, 54, 1)
    return np.expand_dims(resized_spec, axis=0)

# ==========================================
# 5. MAIN LOOP (Optimized for Speed)
# ==========================================
print("\n[LISTENING] Press Ctrl+C to stop...")

# Pre-allocate buffer logic for speed
samples = []
s_append = samples.append # Optimization from your code

try:
    while True:
        # --- A. FAST RECORDING ---
        # Clear buffer
        samples = []
        s_append = samples.append
        
        start_time = time.time()
        
        # Record exactly 3440 samples (4 seconds worth)
        while len(samples) < BUFFER_SIZE:
            s_append(chan.voltage)

        rec_duration = time.time() - start_time
        
        # --- B. PREDICTION ---
        # 1. Preprocess
        input_tensor = preprocess_live_chunk(samples)
        
        # 2. Predict (Verbose=0 hides tensorflow logs)
        predictions = model.predict(input_tensor, verbose=0)
        
        # 3. Result
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        result_label = CLASSES[class_idx]
        
        # Print Status
        print(f"Result: {result_label.upper()} ({confidence:.1f}%) | Rec Time: {rec_duration:.2f}s")
        
except KeyboardInterrupt:
    print("\nStopping...")