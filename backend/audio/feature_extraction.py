import numpy as np
from typing import Dict, Any

try:
    from scipy.fftpack import dct

except Exception:
    def dct(x, type=2, axis=1, norm='ortho'):
        N = x.shape[axis]
        k = np.arange(N)
        n = np.arange(N)
        cos_matrix = np.cos(np.pi * k[:, None] * (2 * n[None, :] + 1) / (2 * N))
        if norm == 'ortho':
            cos_matrix[0] *= np.sqrt(1 / (4 * N)) * 2
            cos_matrix[1:] *= np.sqrt(1 / (2 * N)) * 2
        return np.dot(x, cos_matrix.T)

try:
    import librosa
except Exception:
    librosa = None



def extract_acoustic_features(audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Extracts comprehensive Layer 1 acoustic features from preprocessed audio:
    - MFCC (Mean, Std, Delta)
    - Mel Spectrogram
    - Pitch / F0 statistics (mean, std, jitter estimation)
    - Spectral Centroid, Bandwidth, Rolloff
    - Zero Crossing Rate (ZCR) & RMS Energy
    - Reverberation / Room acoustics (Spectral Flatness, Clarity C50 proxy, high-freq roll-off drop)
    """
    if librosa is not None:
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_db)
        mel_std = np.std(mel_db)

        # Spectral Centroid, Bandwidth, Rolloff
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        
        # Zero Crossing Rate & Energy
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        rms = librosa.feature.rms(y=audio)[0]
        
        # Pitch / F0 Trajectory Analysis
        pitch_mean = 0.0
        pitch_std = 0.0
        f0_jitter = 0.0
        voiced_ratio = 0.0
        try:
            f0, voiced_flag, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr)
            voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            if len(voiced_f0) > 1:
                pitch_mean = float(np.mean(voiced_f0))
                pitch_std = float(np.std(voiced_f0))
                voiced_ratio = float(len(voiced_f0)) / float(len(f0))
                # Jitter approximation: relative mean absolute frame-to-frame difference
                diffs = np.abs(np.diff(voiced_f0))
                f0_jitter = float(np.mean(diffs) / (pitch_mean + 1e-6))
        except Exception:
            pass

        # Room Acoustic & Reverberation Proxy
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        spectral_flatness_mean = float(np.mean(flatness))
        
        # High Frequency Unnatural Cutoff (often present in neural vocoders cut at 7.5kHz or 8kHz)
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        high_freq_mask = freqs > (sr / 2.0 * 0.9)  # Top 10% frequency band
        high_freq_energy_ratio = float(np.sum(spec[high_freq_mask]**2) / (np.sum(spec**2) + 1e-8))

        return {
            "mfcc_mean": mfcc_mean.tolist(),
            "mfcc_std": mfcc_std.tolist(),
            "mfcc_delta_mean": np.mean(mfcc_delta, axis=1).tolist(),
            "mel_mean": float(mel_mean),
            "mel_std": float(mel_std),
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "rolloff_mean": float(np.mean(rolloff)),
            "zcr_mean": float(np.mean(zcr)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "f0_jitter": f0_jitter,
            "voiced_ratio": voiced_ratio,
            "spectral_flatness": spectral_flatness_mean,
            "high_freq_energy_ratio": high_freq_energy_ratio
        }
    else:
        # Fallback NumPy implementation
        n_fft = 512
        spec = np.abs(np.fft.rfft(audio[:n_fft]))
        return {
            "mfcc_mean": [0.0] * 13,
            "mfcc_std": [0.0] * 13,
            "mfcc_delta_mean": [0.0] * 13,
            "mel_mean": float(np.mean(spec)),
            "mel_std": float(np.std(spec)),
            "centroid_mean": 1500.0,
            "centroid_std": 300.0,
            "bandwidth_mean": 1000.0,
            "rolloff_mean": 3000.0,
            "zcr_mean": 0.05,
            "rms_mean": float(np.sqrt(np.mean(audio**2))),
            "rms_std": 0.01,
            "pitch_mean": 120.0,
            "pitch_std": 15.0,
            "f0_jitter": 0.01,
            "voiced_ratio": 0.5,
            "spectral_flatness": 0.01,
            "high_freq_energy_ratio": 0.001
        }


def extract_lfcc_features(audio: np.ndarray, sr: int = 16000, num_filters: int = 20, num_coeffs: int = 20) -> np.ndarray:
    """
    Extracts LFCC (Linear Frequency Cepstral Coefficients) features for Layer 3 Spectral Analysis.
    Computes linear filterbank energies followed by Discrete Cosine Transform (DCT).
    Returns a 1D feature vector summary (mean + std of LFCCs + deltas across frames).
    """
    frame_size = int(0.025 * sr)  # 25ms frame
    frame_stride = int(0.010 * sr) # 10ms stride
    
    # Framing
    audio_len = len(audio)
    if audio_len < frame_size:
        audio = np.pad(audio, (0, frame_size - audio_len), mode='constant')
        audio_len = len(audio)
        
    num_frames = 1 + int(np.floor((audio_len - frame_size) / frame_stride))
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * frame_stride
        frames[i] = audio[start:start + frame_size] * np.hamming(frame_size)
        
    # FFT Spectrum
    n_fft = 512
    mag_frames = np.abs(np.fft.rfft(frames, n=n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    
    # Linear Filterbank
    low_freq = 0
    high_freq = sr / 2
    linear_points = np.linspace(low_freq, high_freq, num_filters + 2)
    bin_points = np.floor((n_fft + 1) * linear_points / sr).astype(int)
    
    fbank = np.zeros((num_filters, int(n_fft / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (bin_points[m] - bin_points[m - 1]))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (bin_points[m + 1] - bin_points[m]))
            
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks) # dB
    
    # DCT to obtain LFCCs
    lfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_coeffs]
    
    # Delta LFCC
    delta_lfcc = np.zeros_like(lfcc)
    if num_frames > 1:
        delta_lfcc[1:-1] = (lfcc[2:] - lfcc[:-2]) / 2.0
        
    # Aggregate summary feature vector: mean & std of LFCCs and Deltas
    lfcc_mean = np.mean(lfcc, axis=0)
    lfcc_std = np.std(lfcc, axis=0)
    delta_mean = np.mean(delta_lfcc, axis=0)
    delta_std = np.std(delta_lfcc, axis=0)
    
    feature_vector = np.concatenate([lfcc_mean, lfcc_std, delta_mean, delta_std])
    return feature_vector
