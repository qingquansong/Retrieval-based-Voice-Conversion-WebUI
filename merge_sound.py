import numpy as np
import librosa
from pydub import AudioSegment

def _dtype_from_width(width):
    # helper to map sample_width → numpy dtype
    return {1: np.int8, 2: np.int16, 4: np.int32}[width]

def pitch_shift(sound: AudioSegment, semitones: float) -> AudioSegment:
    """
    Shift pitch by `semitones` (e.g. +12 = up one octave, -12 = down one octave)
    WITHOUT changing duration—uses librosa’s phase‐vocoder internally.
    """
    # 1) pull raw samples into a numpy array
    samples = np.array(sound.get_array_of_samples())
    if sound.channels > 1:
        # pydub interleaves channels in the array; reshape to (channels, n_samples)
        samples = samples.reshape((-1, sound.channels)).T
    # convert to float32 in [-1,1]
    max_val = float(1 << (8 * sound.sample_width - 1))
    samples = samples.astype(np.float32) / max_val

    # 2) do the pitch‐shift
    y_shifted = librosa.effects.pitch_shift(samples,
                                            sr=sound.frame_rate,
                                            n_steps=semitones)

    # 3) bring back to int PCM
    if sound.channels > 1:
        # transpose back and flatten to interleaved
        y_shifted = y_shifted.T.flatten()
    y_int = np.clip(y_shifted * max_val,
                    -max_val, max_val - 1).astype(_dtype_from_width(sound.sample_width))

    # 4) build a new AudioSegment
    return AudioSegment(
        data=y_int.tobytes(),
        sample_width=sound.sample_width,
        frame_rate=sound.frame_rate,
        channels=sound.channels
    )


# --- load ---
instrument = AudioSegment.from_wav("/home/jobuser/1_AtHeart_(Instrumental).wav")
vocals     = AudioSegment.from_wav("/home/jobuser/1_vocal_vocal_with_harmony_reverb.wav")

# # --- pitch-shift examples ---
# # raise instrument by 1 octave
# instr_up1 = pitch_shift(instrument, 1.0)

# # lower vocals by a perfect fifth (1 octave = 12 semitones; perfect 5th ≈ 7 semitones → 7/12 octave)
# vocals_down5th = pitch_shift(vocals, -7/12)

# --- shift only pitch +1 semitone (≈1/12 octave) ---
# processed_instr = pitch_shift(instrument, 0.0)
processed_instr = instrument

# # --- volume (gain) examples ---
# # boost vocals by -1 dB
# processed_vocals = vocals - 1

# # cut instrument by +1 dB
# processed_instr = processed_instr + 1

# # --- combine everything ---
# # e.g. take the octave-up instrument, drop it by 6 dB, overlay vocals (shifted & boosted)
# processed_instr = pitch_shift(instrument, 1.0) - 6
# processed_vocals = pitch_shift(vocals, 0.0) + 3

processed_vocals = vocals # + 15


mixed = processed_instr.overlay(processed_vocals)

mixed.export("atheart_yichun_with_harmony_reverb.wav", format="wav")
