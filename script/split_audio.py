import os
from tqdm import tqdm
import glob
import torch
import torchaudio
import pickle
import sys
from extract_audiomae_embedding import silence_detection
sys.path.append('../')

# EXP params
split_duration = 1
split_hop = split_duration
split_pkl_path = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet_0.1_threshold_pkl/'
src_audio_path = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/'

# Global vars
audio_sample_rate=22050
audio_len = 4

def split_and_save_audio_into_pkl(audio_wavs, split_duration, 
                                  split_hop, save_pkl_path, 
                                  silence_threshold=0.1):

    # split supports 1 to 4 sec
    num_splits = int((audio_len - split_duration) // split_hop + 1)

    for audio_wav in tqdm(audio_wavs):
        audio_signal, sr = torchaudio.load(audio_wav)
        audio_class = audio_wav.split('/')[-2]

        audio_class_pkl_path = os.path.join(save_pkl_path, audio_class)

        os.makedirs(audio_class_pkl_path, exist_ok=True)

        # normalize by maximum value
        max_value = torch.max(torch.abs(audio_signal))
        assert max_value > 0
        audio_signal_norm = audio_signal / max_value

        # using moving window to process audio segments
        for split_idx in range(num_splits):
            sample_idx_start = split_idx * split_hop * audio_sample_rate
            sample_idx_end = split_idx * split_hop * audio_sample_rate + split_duration * audio_sample_rate

            audio_seg = audio_signal[:, int(sample_idx_start): int(sample_idx_end)]
            audio_seg_norm = audio_signal_norm[:, int(sample_idx_start): int(sample_idx_end)]

            # discard if silence
            if silence_detection(audio_seg_norm, threshold=silence_threshold):
                continue

            audio_pkl_split_path = os.path.join(audio_class_pkl_path, 
                                                audio_wav.split('/')[-1].split('.')[0] + '_' + str(split_idx) + '.pkl')

            with open(audio_pkl_split_path, 'wb') as pickle_file:
                pickle.dump(audio_seg, pickle_file)

if __name__ == "__main__":

    audio_wavs = glob.glob(os.path.join(src_audio_path, '**/*.wav'), recursive=True)
    split_and_save_audio_into_pkl(audio_wavs, split_duration=split_duration,
                                  split_hop=split_hop, save_pkl_path=split_pkl_path)
