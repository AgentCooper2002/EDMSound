import os
from tqdm import tqdm
import glob
import torch
import torchaudio
from torchaudio.compliance import kaldi
from torch.nn import functional as F
import pickle
from panns_inference import AudioTagging
import numpy as np


# EXP params
split_duration = 4
split_hop = split_duration / 2
extract_obj = 'GEN'
pkl_path = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/'
audio_path = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/' if extract_obj == 'GT' else '/home/yutong/5/tensorboard/test_samples/'

# Global vars
device = torch.device("cuda")
audio_sample_rate=22050
model_sample_rate=32000
audio_len = 4
label_idx_dict = {'DogBark': 0, 'Footstep': 1, 'GunShot': 2, 
                  'Keyboard': 3, 'MovingMotorVehicle': 4, 
                  'Rain': 5, 'Sneeze_Cough': 6}

label_idx_dict_dcase_sub = {'dog_bark': 0, 'footstep': 1, 'gunshot': 2, 
                            'keyboard': 3, 'moving_motor_vehicle': 4, 
                            'rain': 5, 'sneeze_cough': 6}

# Resampler
resample_transform = torchaudio.transforms.Resample(audio_sample_rate, 
                                                    model_sample_rate)

# load clap
panns_model = AudioTagging(checkpoint_path=None, device='cuda')


def compute_panns_embed(audio_wavs, split_duration, split_hop, save_pkl_path):

    # split supports 1 to 4 sec
    num_splits = int((audio_len - split_duration) // split_hop + 1)
    clap_vec_dict = {}

    for audio_wav in tqdm(audio_wavs):
        audio_signal, sr = torchaudio.load(audio_wav)

        # resampling if necessary
        if sr != model_sample_rate:
            audio_signal = resample_transform(audio_signal).to(device)

        # using moving window to process audio segments
        audio_matrix = []
        for split_idx in range(num_splits):
            sample_idx_start = split_idx * split_hop * model_sample_rate
            sample_idx_end = split_idx * split_hop * model_sample_rate + split_duration * model_sample_rate

            audio_seg = audio_signal[:, int(sample_idx_start): int(sample_idx_end)]
            
            # avoid naming problems
            if 'DCASE' in audio_wav:
                class_name = audio_wav.split('/')[-2]
                class_label = label_idx_dict[class_name]
            elif 'Submissions' in audio_wav:
                class_name = audio_wav.split('/')[-2]
                class_label = label_idx_dict_dcase_sub[class_name]
            else:
                class_label = int(audio_wav.split('/')[-1].split('_')[1])
                
            with torch.no_grad():
                (_, audio_vector) = panns_model.inference(audio_seg)
            audio_matrix.append(audio_vector)
        clap_vec_dict[audio_wav] = {'vector': np.array(audio_matrix), 'label':class_label}

    with open(save_pkl_path, 'wb') as pickle_file:
        pickle.dump(clap_vec_dict, pickle_file)

if __name__ == "__main__":

    embed_pkl_path = os.path.join(pkl_path, extract_obj + '_panns_EDMSound_' + str(split_duration) + '.pkl')
    # audio_wavs = glob.glob(os.path.join(audio_path, '**/*.wav'), recursive=True)
    audio_wavs = glob.glob(os.path.join(audio_path, '*.wav'), recursive=True)   
    compute_panns_embed(audio_wavs, split_duration=split_duration,
                           split_hop=split_hop, save_pkl_path=embed_pkl_path)
