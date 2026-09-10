import os
import numpy as np
from typing import Tuple, Dict, Any
from fastapi import HTTPException, status
from backend.config import settings

# Attempt PyAV import for universal audio/video container decoding
try:
    import av
except Exception:
    av = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    import librosa
except Exception:
    librosa = None


def decode_with_pyav(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int, int, float]:
    """
    Decodes ANY audio or video container format (AAC, M4A, WMA, OPUS, AMR, MP3, WAV, OGG, FLAC, MP4, etc.)
    into a mono float32 numpy array resampled to target_sr using PyAV.
    """
    container = av.open(file_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        raise ValueError("No audio stream found in media file.")

    resampler = av.AudioResampler(
        format='flt', # 32-bit float
        layout='mono',
        rate=target_sr
    )

    audio_samples = []
    orig_sr = audio_stream.codec_context.sample_rate or target_sr
    orig_channels = audio_stream.codec_context.channels or 1

    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        for r_frame in resampled_frames:
            # Convert frame buffer to numpy float32 array
            arr = r_frame.to_ndarray()
            audio_samples.append(arr.flatten())

    container.close()

    if not audio_samples:
        raise ValueError("PyAV extracted zero audio samples.")

    audio_data = np.concatenate(audio_samples).astype(np.float32)
    duration = float(len(audio_data)) / float(target_sr)

    return audio_data, target_sr, orig_channels, duration


def load_and_preprocess_audio(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads ANY audio file format, standardizes to 16kHz mono WAV float32 array,
    normalizes amplitude, and returns audio data along with metadata profile.
    """
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Specified audio file does not exist."
        )

    ext = os.path.splitext(file_path)[1].lstrip('.').lower() or "wav"
    audio_data = None
    sr = settings.TARGET_SAMPLE_RATE
    orig_channels = 1
    duration = 0.0

    # 1. Try PyAV first (Universal decoder for AAC, M4A, WMA, OPUS, MP3, WAV, MP4, FLAC, etc.)
    if av is not None:
        try:
            audio_data, sr, orig_channels, duration = decode_with_pyav(file_path, target_sr=settings.TARGET_SAMPLE_RATE)
        except Exception as pyav_err:
            audio_data = None

    # 2. Fallback to librosa
    if audio_data is None and librosa is not None:
        try:
            audio_data, sr = librosa.load(file_path, sr=settings.TARGET_SAMPLE_RATE, mono=True)
            duration = float(len(audio_data)) / float(sr)
            if sf is not None:
                try:
                    info = sf.info(file_path)
                    orig_channels = info.channels
                except Exception:
                    pass
        except Exception:
            audio_data = None

    # 3. Fallback to soundfile
    if audio_data is None and sf is not None:
        try:
            data, orig_sr = sf.read(file_path)
            orig_channels = data.shape[1] if data.ndim > 1 else 1
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if orig_sr != settings.TARGET_SAMPLE_RATE:
                num_samples = int(len(data) * settings.TARGET_SAMPLE_RATE / orig_sr)
                audio_data = np.interp(
                    np.linspace(0, len(data), num_samples, endpoint=False),
                    np.arange(len(data)),
                    data
                )
                sr = settings.TARGET_SAMPLE_RATE
            else:
                audio_data = data
                sr = orig_sr
            duration = float(len(audio_data)) / float(sr)
        except Exception:
            audio_data = None

    # 4. Fallback to pydub
    if audio_data is None:
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_file(file_path)
            orig_channels = sound.channels
            sound = sound.set_channels(1).set_frame_rate(settings.TARGET_SAMPLE_RATE)
            samples = sound.get_array_of_samples()
            audio_data = np.array(samples, dtype=np.float32) / (1 << (8 * sound.sample_width - 1))
            sr = settings.TARGET_SAMPLE_RATE
            duration = float(len(audio_data)) / float(sr)
        except Exception as py_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read audio file '{file_path}'. Format may be corrupted or unsupported. Error: {str(py_err)}"
            )

    if audio_data is None or len(audio_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Loaded audio file contained zero audio samples."
        )

    # Duration Checks
    if duration < settings.MIN_AUDIO_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration:.2f}s) is too short. Minimum duration is {settings.MIN_AUDIO_DURATION_SEC}s."
        )
    if duration > settings.MAX_AUDIO_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration ({duration:.2f}s) exceeds maximum limit of {settings.MAX_AUDIO_DURATION_SEC}s."
        )

    # Peak Normalization
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    metadata = {
        "duration": round(duration, 2),
        "format": ext,
        "sample_rate": sr,
        "channels": orig_channels,
        "num_samples": len(audio_data)
    }

    return audio_data, metadata
