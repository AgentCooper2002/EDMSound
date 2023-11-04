import os
from tqdm import tqdm
import glob
import torch
import torchaudio
from torchaudio.compliance import kaldi
from torch.nn import functional as F
import pickle
import sys
import numpy as np
sys.path.append('../')
from src.models.components.backbones.audioMAE.AudioMAE import models_mae

# EXP params
split_duration = 4
split_hop = split_duration / 2
extract_obj = 'GEN'
pkl_path = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/'
audio_path = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/' if extract_obj == 'GT' else '/home/yutong/5/tensorboard/test_samples/'

# Global vars
device = torch.device("cuda")
audio_sample_rate=22050
model_sample_rate=16000
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

# load audiomae
mae_model = models_mae.__dict__['mae_vit_base_patch16'](in_chans=1, 
                                                        audio_exp=True,
                                                        img_size=(1024, 128),
                                                        decoder_mode=1,
                                                        decoder_depth=16)
checkpoint = torch.load('/storageNVME/yutong/pretrained.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
mae_model.load_state_dict(checkpoint_model)
mae_model.to(device)
mae_model.eval()

def silence_detection(audio, threshold=0.03):
    # # need to tune threshold

    peak, _ = torch.max(torch.abs(audio), dim=-1)
    return peak < threshold

def extract_fbank(audio, process_len):

    if audio.shape[-1] > process_len:
        start = 0 # random.randint(0, audio.shape[-1] - self.audio_len)
        end = start + process_len
        audio = audio[:, start:end]
    elif audio.shape[-1] < process_len:
        pad = max(process_len - audio.shape[-1], 0)
        audio = F.pad(audio, (pad//2, pad//2+(pad%2)), mode='constant')
    
    return to_fbank(audio, model_sample_rate)

def to_fbank(audio, sample_rate, target_len=1024):

    for i in range(audio.shape[0]):
        chunk = kaldi.fbank(audio[i].unsqueeze(0), htk_compat=True, 
                            sample_frequency=sample_rate, 
                            use_energy=False, 
                            window_type='hanning', 
                            num_mel_bins=128, 
                            dither=0.0, frame_shift=10).unsqueeze(0)
        if i == 0:
            audio_feat = chunk
        else:
            audio_feat = torch.cat((audio_feat, chunk), dim=0)
    
    n_frames = audio_feat.shape[1]    
    p = target_len - n_frames
 
    if p > 0:
        n = target_len // n_frames + 1      
        audio_feat = audio_feat.repeat(1, n, 1)
             
    n_frames = audio_feat.shape[1]
    p = target_len - n_frames
    if p < 0:
        audio_feat = audio_feat[:, :target_len, :]

    return audio_feat

def compute_audiomae_embed(audio_wavs, split_duration, split_hop, save_pkl_path, threshold=0.3):

    # split supports 1 to 4 sec
    num_splits = int((audio_len - split_duration) // split_hop + 1)
    audiomae_vec_dict = {}

    for audio_wav in tqdm(audio_wavs):
        audio_signal, sr = torchaudio.load(audio_wav)

        # resampling if necessary
        if sr != model_sample_rate:
            audio_signal = resample_transform(audio_signal).to(device)

        # normalize by maximum value
        max_value = torch.max(torch.abs(audio_signal))
        assert max_value > 0
        audio_signal_norm = audio_signal / max_value

        # using moving window to process audio segments
        audio_matrix = []
        for split_idx in range(num_splits):
            sample_idx_start = split_idx * split_hop * model_sample_rate
            sample_idx_end = split_idx * split_hop * model_sample_rate + split_duration * model_sample_rate

            audio_seg = audio_signal[:, int(sample_idx_start): int(sample_idx_end)]
            audio_seg_norm = audio_signal_norm[:, int(sample_idx_start): int(sample_idx_end)]

            # discard if silence
            if silence_detection(audio_seg_norm, threshold):
                audio_matrix.append(np.full((1, 768), np.inf))
                continue
            
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
                audio_fbank = extract_fbank(audio_seg, process_len=int(split_duration*model_sample_rate))
                audio_vector = mae_model.forward_encoder_no_mask(audio_fbank.unsqueeze(1))
                audio_vector = torch.mean(audio_vector, dim=1)

            audio_matrix.append(audio_vector.cpu().numpy())

        audiomae_vec_dict[audio_wav] = {'vector': np.array(audio_matrix), 'label':class_label}

    with open(save_pkl_path, 'wb') as pickle_file:
        pickle.dump(audiomae_vec_dict, pickle_file)

if __name__ == "__main__":

    embed_pkl_path = os.path.join(pkl_path, extract_obj + '_audiomae_EDMSound_' + str(split_duration) + '.pkl')
    # audio_wavs = glob.glob(os.path.join(audio_path, '**/*.wav'), recursive=True)
    audio_wavs = glob.glob(os.path.join(audio_path, '*.wav'), recursive=True)

    compute_audiomae_embed(audio_wavs, split_duration=split_duration,
                           split_hop=split_hop, save_pkl_path=embed_pkl_path)
