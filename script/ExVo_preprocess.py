import os
import soundfile
import librosa
from scipy.io import wavfile
import noisereduce as nr
import numpy as np
from tqdm import tqdm
import glob

def get_wav(file_name, wav_path):
    """ Loads wav dat at index

    :param file_name: File name
    :param wav_path: Base path of wavs
    :return: (np.array of wav data, sr)
    """
    path = os.path.join(wav_path, file_name)
    data, sr = soundfile.read(path)

    # If audio data has 2 channel, just take average of both
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr

def preprocess_audio(wav, sr=16000):
    """ Removes noise from .wav data using noisereduce

    :param wav: np.array of .wav data
    :param sr: Sample rate of sample
    :return: np.array of denoised audio
    """
    trimmed = np.array([])
    reduced_noise = nr.reduce_noise(y=wav, sr=sr)
    # trimmed, _ = librosa.effects.trim(reduced_noise, top_db=30)
    non_silence = librosa.effects.split(y=reduced_noise, top_db=40)
    for i in range(len(non_silence)):
        start, end = non_silence[i]
        if i == 0:
            trimmed = reduced_noise[start:end]
        else:
            trimmed = np.concatenate((trimmed, reduced_noise[start:end]))
    return trimmed

if __name__ == "__main__":

    new_path = '/storageNVME/yutong/hume-vocalbursts/2022_ACII_A-VB/preprocessed_trim3/'
    os.makedirs(new_path, exist_ok=True)

    count = 0
    message = []
    for file_name in tqdm(glob.glob("/storageNVME/yutong/hume-vocalbursts/2022_ACII_A-VB/audio/wav/*.wav", recursive=True)):
        data, sr = get_wav(file_name, '/storageNVME/yutong/hume-vocalbursts/2022_ACII_A-VB/audio/wav/')
        data = preprocess_audio(data, sr)
        length = data.shape[-1]/sr
        file_name = file_name.split('/')[-1]
        filtered_file = os.path.join(new_path, file_name)
        if length < 0.05:
            count += 1
            message.append(f'length, {file_name}')
            continue
        wavfile.write(filtered_file, sr, data)
