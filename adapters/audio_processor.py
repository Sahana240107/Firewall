import numpy as np
import soundfile as sf
import subprocess
import os

BLEEP_FREQUENCY_HZ = 1000
BLEEP_VOLUME_DB = -10
BLEEP_PADDING_MS = 100


def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert any audio format to wav using imageio-ffmpeg."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", input_path, output_path],
        check=True, capture_output=True
    )
    return output_path

def apply_bleeps(input_path: str, output_path: str, segments: list) -> None:
    """
    Replace each sensitive time range with a 1kHz bleep tone.
    segments: list of dicts with 'start' and 'end' in seconds.
    Skips segments with start == -1 (timestamp not found).
    """
    # Convert to wav first if needed
    ext = os.path.splitext(input_path)[1].lower()
    if ext != ".wav":
        wav_path = input_path.replace(ext, "_converted.wav")
        convert_to_wav(input_path, wav_path)
        input_path = wav_path

    audio, sample_rate = sf.read(input_path, always_2d=True)
    bleep_amplitude = 10 ** (BLEEP_VOLUME_DB / 20) * 0.5
    audio_duration_ms = (len(audio) / sample_rate) * 1000

    bleeped_count = 0

    for seg in segments:
        # Skip segments without valid timestamps
        if seg.get("start", -1) == -1 or seg.get("end", -1) == -1:
            print(f"[Bleep] Skipping '{seg.get('text', '')}' — no timestamp")
            continue

        start_ms = seg["start"] * 1000
        end_ms = seg["end"] * 1000

        # Add padding
        padded_start_ms = max(0, start_ms - BLEEP_PADDING_MS)
        padded_end_ms = min(audio_duration_ms, end_ms + BLEEP_PADDING_MS)

        start_sample = int((padded_start_ms / 1000) * sample_rate)
        end_sample = int((padded_end_ms / 1000) * sample_rate)
        duration_samples = end_sample - start_sample

        if duration_samples <= 0:
            continue

        # Generate 1kHz sine wave bleep
        t = np.linspace(
            0,
            duration_samples / sample_rate,
            duration_samples,
            endpoint=False
        )
        bleep_mono = (
            np.sin(2 * np.pi * BLEEP_FREQUENCY_HZ * t) * bleep_amplitude
        ).astype(audio.dtype)

        # Match audio channels
        num_channels = audio.shape[1]
        bleep = np.stack([bleep_mono] * num_channels, axis=1)

        # Splice bleep into audio
        audio[start_sample:end_sample] = bleep[:duration_samples]

        print(f"[Bleep] ✅ {seg.get('type', 'UNKNOWN')} at {seg['start']}s–{seg['end']}s")
        bleeped_count += 1

    sf.write(output_path, audio, sample_rate)
    print(f"[Done] {bleeped_count} segment(s) bleeped → {output_path}")
