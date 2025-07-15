import numpy as np
import librosa
import soundfile as sf
import os
import random

# F0 estimation using librosa's pyin
def estimate_f0(wav, fs):
    f0, voiced_flag, voiced_probs = librosa.pyin(wav, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0

# Spectral envelope estimation using librosa's melspectrogram
def estimate_spectral_envelope(wav, fs, n_fft=1024, n_mels=80):
    S = np.abs(librosa.stft(wav, n_fft=n_fft))
    mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels)
    mel_S = np.dot(mel_basis, S**2)
    return mel_S

# Aperiodic component estimation (dummy implementation)
def estimate_aperiodic_component(wav, fs):
    return np.zeros_like(wav)

# Synthesis using inverse short-time Fourier transform (ISTFT)
def synthesize(f0, spectral_envelope, aperiodic_component, fs, hop_length=256):
    n_frames = spectral_envelope.shape[1]
    n_samples = n_frames * hop_length
    y = np.zeros(n_samples)
    for t in range(n_frames):
        spectrum = np.random.randn(spectral_envelope.shape[0]) * np.sqrt(spectral_envelope[:, t])
        y_frame = librosa.istft(spectrum, hop_length=hop_length)
        y[t * hop_length: (t + 1) * hop_length] += y_frame
    return y

# Main function to process a single audio file
def process_file(wav_path, output_path):
    wav, fs = sf.read(wav_path)
    f0 = estimate_f0(wav, fs)
    spectral_envelope = estimate_spectral_envelope(wav, fs)
    aperiodic_component = estimate_aperiodic_component(wav, fs)
    synthesized_wav = synthesize(f0, spectral_envelope, aperiodic_component, fs)
    sf.write(output_path, synthesized_wav, fs)

# Directory paths
wav_scp = '/Work29/wwm1995/SMIIP/Anti_Spoof/ASVspoof5/data/train_bona/bona.scp'
output_dir = '/Corpus3/yikang/ASVspoof5/train_world/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    with open(wav_scp, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        utt_id, wav_path = line.split()
        output_path = os.path.join(output_dir, utt_id + '.wav')
        process_file(wav_path, output_path)

def debug_main():
    with open(wav_scp, 'r') as f:
        lines = f.readlines()

    # For debugging, we will use a small sample of lines
    sample_size = 5  # Number of samples to use for debugging
    lines = random.sample(lines, sample_size)

    for line in lines:
        line = line.strip()
        utt_id, wav_path = line.split()
        print(f"Processing {utt_id} for debugging...")
        process_file(wav_path, "/dev/null")

if __name__ == '__main__':
    debug = True  # Change to False to run the main function
    if debug:
        debug_main()
    else:
        main()
