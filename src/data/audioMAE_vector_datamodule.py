from typing import Any, Dict, Optional, Tuple
import torch
import glob, random, os
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchaudio.compliance import kaldi
import torchaudio.functional as F
import torchaudio.transforms as T
import torch.nn as nn


from ..models.components.backbones.audioMAE.AudioMAE import models_mae
import pickle

def to_fbank_aug(audio, sample_rate, target_len=1024, aug=True):
    audio = kaldi.fbank(audio, 
                        htk_compat=True, 
                        sample_frequency=sample_rate, 
                        use_energy=False, 
                        window_type='hanning', 
                        num_mel_bins=128, 
                        dither=0.0, 
                        frame_shift=10)
    # audio = get_spectrogram(audio)
        # print(n_frames)
    if random.random() < 0.9 and aug:
        audio = audio.T
        audio = audio.unsqueeze(0)
        percentage_time = random.random() * 0.1
        percentage_freq = random.random() * 0.1
        time_beam = int(percentage_time * audio.shape[-1])
        freq_beam = int(percentage_freq * audio.shape[1])
        time_masking = T.TimeMasking(time_mask_param=time_beam)
        freq_masking = T.FrequencyMasking(freq_mask_param=freq_beam)
        audio = time_masking(audio)
        audio = freq_masking(audio)
        audio = audio.squeeze(0).T
        
    n_frames = audio.shape[0]    
    p = target_len - n_frames
 
    if p > 0:
        n = target_len // n_frames + 1      
        audio = audio.repeat(n, 1)
            
    n_frames = audio.shape[0]
    p = target_len - n_frames
    if p < 0:
        audio = audio[:target_len, :]
    return audio

class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, time = wav.size()
        device = wav.device
        if random.random() < self.proba:
            scales = torch.empty(batch, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav

class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """
    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        shift = int(random.random() * self.shift)
        batch, sources, channels, time = wav.size()
        length = time - shift
        if shift > 0: 
            srcs = 1 if self.same else sources
            offsets = torch.randint(shift, [batch, srcs, 1, 1], device=wav.device)
            offsets = offsets.expand(-1, sources, channels, -1)
            indexes = torch.arange(length, device=wav.device)
            wav = wav.gather(3, indexes + offsets)
            if time > wav.shape[-1]:
                padding_size = time - wav.shape[-1]
                padding = (0, padding_size)
                wav = torch.nn.functional.pad(wav, padding, "constant", value=0)
        return wav

class AudioMAEFineTuneDataset(Dataset):
    def __init__(self, path, 
                 target_sample_rate,
                 data_sample_rate, 
                 pkl_audio=False,
                 return_pairs=True, 
                 mixup_m = 0.3):
        super().__init__()
        self.pkl_audio = pkl_audio
        self.return_pairs = return_pairs
        self.mixup_m = mixup_m

        # resampler
        self.resample = True if target_sample_rate != data_sample_rate else False
        if self.resample:
            self.resample_transform = torchaudio.transforms.Resample(data_sample_rate, 
                                                                     target_sample_rate)

        # label to idx
        self.label_idx_dict = {'DogBark': 0, 'Footstep': 1, 'GunShot': 2, 
                               'Keyboard': 3, 'MovingMotorVehicle': 4, 
                               'Rain': 5, 'Sneeze_Cough': 6}

        # get all filenames based on the input format
        if pkl_audio:
            self.filenames = glob.glob(f'{path}/**/*.pkl', recursive=True)
            audio_suffix = '.pkl'
        else:
            self.filenames = glob.glob(f'{path}/**/*.wav', recursive=True)
            audio_suffix = '.wav'

        # get all filenames for each class
        self.class_filenames = []
        for class_id in self.label_idx_dict.keys():
            class_filenames = glob.glob(os.path.join(path, class_id, '*' + audio_suffix), recursive=True)
            self.class_filenames.append(class_filenames)
        random.seed(42)
        
    def __len__(self):
        return len(self.filenames)
    
    def get_audio(self, audio_filename):
        if self.pkl_audio:
            # this is only used in split case
            with open(audio_filename, 'rb') as f:
                signal = pickle.load(f)
        else:
            signal, _ = torchaudio.load(audio_filename)
        return signal
    
    def __getitem__(self, idx):
        
        audio_fn_a = self.filenames[idx]
        signal_a = self.get_audio(audio_fn_a)

        # read class label
        class_name = audio_fn_a.split('/')[-2]
        class_label = self.label_idx_dict[class_name]

        # start positive and negative selection
        if self.return_pairs:
            random_select_seed = random.random()
            if random_select_seed < 0.2:
                signal_b = signal_a.clone()
                label = 1.0
            else:

                filenames = self.class_filenames[class_label].copy()
                filenames.remove(audio_fn_a)
                rnd_idx = torch.randint(0, len(filenames), (1,)).item()
                signal_b = self.get_audio(filenames[rnd_idx])
            
            if self.resample:
                signal_a = self.resample_transform(signal_a)
                signal_b = self.resample_transform(signal_b)
            
            return {'audio_a': signal_a[0], 
                    'audio_b': signal_b[0],
                    'label': label}
            
        else:
            # generate TWO random numbers
            if self.mixup_m > 0:
                rn_1 = random.uniform(0.5 + self.mixup_m, 1)
                rn_2 = random.uniform(0, 0.5 - self.mixup_m)
            else:
                rn_1 = 1
                rn_2 = 0

            # generate the second audio
            filenames = self.class_filenames[class_label].copy()
            filenames.remove(audio_fn_a)
            rnd_idx = torch.randint(0, len(filenames), (1,)).item()
            signal_b_ = self.get_audio(filenames[rnd_idx])

            if self.resample:
                signal_a = self.resample_transform(signal_a)
                signal_b_ = self.resample_transform(signal_b_)

            # mixup
            signal_b = signal_a * rn_1 + signal_b_ * (1 - rn_1)
            signal_c = signal_a * rn_2 + signal_b_ * (1 - rn_2)

            return {'audio_a': signal_a[0], 
                    'audio_b': signal_b[0],
                    'audio_c': signal_c[0]}
    
class Collator:
    def __init__(self, audio_len, sample_rate):
        self.audio_len = audio_len
        self.sample_rate = sample_rate
        self.scale = Scale()
        self.shift = Shift(shift=sample_rate//5)
        self.device = torch.device("cuda")
        self.model = models_mae.__dict__['mae_vit_base_patch16'](in_chans=1, 
                                                        audio_exp=True,
                                                        img_size=
                                                        (1024, 128),
                                                        decoder_mode=1,
                                                        decoder_depth=16)
        checkpoint = torch.load('/storageNVME/yutong/pretrained.pth', map_location='cpu')
        checkpoint_model = checkpoint['model']
        _ = self.model.load_state_dict(checkpoint_model)
        self.model.to(self.device)
        
    def prepare(self, audio_wav, transform):
        assert len(audio_wav) >= self.audio_len
        if len(audio_wav) > self.audio_len:
            start = random.randint(0, audio_wav.shape[-1] - self.audio_len)
            end = start + self.audio_len
            audio_wav = audio_wav[start:end]
        if transform:
            snr_db = random.uniform(80, 100)
            noise = torch.randn_like(audio_wav)
            audio_wav = F.add_noise(audio_wav.unsqueeze(0), noise.unsqueeze(0), torch.tensor([snr_db]))
            audio_wav = self.scale(audio_wav)
            audio_wav = self.shift(audio_wav.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return audio_wav
        
    def collate_pair(self, minibatch):
        class_labels = []
        for record in minibatch:

            record['audio_a'] = self.prepare(record['audio_a'], 
                                             transform=False)
            record['audio_b'] = self.prepare(record['audio_b'], 
                                             transform=True)
 
            class_labels.append(record['label'])
            
            record['wav_a'] = record['audio_a'].clone()
            record['wav_b'] = record['audio_b'].squeeze(0).clone()  
            
            record['audio_b'] = to_fbank_aug(record['audio_b'], 
                                             self.sample_rate)
            record['audio_a'] = to_fbank_aug(record['audio_a'].unsqueeze(0), 
                                             self.sample_rate, aug=False)
            
        # and then to_mel using the fbank function from audioMAE 
        audio_a = torch.stack([record['audio_a'] for record in minibatch]).unsqueeze(1).to(self.device)
        audio_b = torch.stack([record['audio_b'] for record in minibatch]).unsqueeze(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            audio_a = self.model.forward_encoder_no_mask(audio_a)
            audio_b = self.model.forward_encoder_no_mask(audio_b)
        audio_a = torch.mean(audio_a, dim=1)
        audio_b = torch.mean(audio_b, dim=1)
        wav_a = np.stack([record['wav_a'] for record in minibatch])
        wav_b = np.stack([record['wav_b'] for record in minibatch])
        
        return {'audio_a': audio_a,
                'audio_b': audio_b,
                'label': torch.tensor(class_labels),
                'wav_a': torch.from_numpy(wav_a),
                'wav_b': torch.from_numpy(wav_b)}
        
    def collate_triplet(self, minibatch):

        for record in minibatch:

            record['audio_a'] = self.prepare(record['audio_a'], transform=False)
            record['audio_b'] = self.prepare(record['audio_b'], transform=True)
            record['audio_c'] = self.prepare(record['audio_c'], transform=True)
 
            record['wav_a'] = record['audio_a'].clone()
            record['wav_b'] = record['audio_b'].squeeze(0).clone()  
            record['wav_c'] = record['audio_c'].squeeze(0).clone()  
            
            record['audio_b'] = to_fbank_aug(record['audio_b'], self.sample_rate)
            record['audio_c'] = to_fbank_aug(record['audio_c'], self.sample_rate)
            record['audio_a'] = to_fbank_aug(record['audio_a'].unsqueeze(0), self.sample_rate, aug=False)
            
        # and then to_mel using the fbank function from audioMAE 
        audio_a = torch.stack([record['audio_a'] for record in minibatch]).unsqueeze(1).to(self.device)
        audio_b = torch.stack([record['audio_b'] for record in minibatch]).unsqueeze(1).to(self.device)
        audio_c = torch.stack([record['audio_c'] for record in minibatch]).unsqueeze(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            audio_a = self.model.forward_encoder_no_mask(audio_a)
            audio_b = self.model.forward_encoder_no_mask(audio_b)
            audio_c = self.model.forward_encoder_no_mask(audio_c)
        audio_a = torch.mean(audio_a, dim=1)
        audio_b = torch.mean(audio_b, dim=1)
        audio_c = torch.mean(audio_c, dim=1)
        wav_a = np.stack([record['wav_a'] for record in minibatch])
        wav_b = np.stack([record['wav_b'] for record in minibatch])
        wav_c = np.stack([record['wav_c'] for record in minibatch])
        
        return {'audio_a': audio_a,
                'audio_b': audio_b,
                'audio_c': audio_c,
                'wav_a': torch.from_numpy(wav_a),
                'wav_b': torch.from_numpy(wav_b),
                'wav_c': torch.from_numpy(wav_c)}

class AudioMAEFineTuneDataModule(LightningDataModule):
    """A DataModule implements 5 key methods:
    
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "./",
        audio_len: int = 16000,
        split: int = 1,
        target_sample_rate: int = 16000,
        data_sample_rate: int = 22050,
        num_class: int = 10,
        return_pairs: bool = True,
        mixup_m: float = 0.1,
        train_val_split: Tuple[float, float] = (0.9, 0.05),
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.audio_len = audio_len // split
        self.pkl_audio = False
        if split > 1:
            self.pkl_audio = True
        

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return self.hparams.num_class

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        dataset = AudioMAEFineTuneDataset(self.hparams.data_dir, 
                                          target_sample_rate=self.hparams.target_sample_rate,
                                          data_sample_rate=self.hparams.data_sample_rate, 
                                          return_pairs=self.hparams.return_pairs,
                                          pkl_audio=self.pkl_audio,
                                          mixup_m=self.hparams.mixup_m)
        
        train_len = int(self.hparams.train_val_split[0]*len(dataset))
        val_len = int(self.hparams.train_val_split[1]*len(dataset))
        test_len = len(dataset) - train_len - val_len
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=[train_len, val_len, test_len],
                    generator=torch.Generator().manual_seed(42))


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(self.audio_len, 
                                sample_rate=self.hparams.target_sample_rate).collate_triplet,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(self.audio_len, 
                                sample_rate=self.hparams.target_sample_rate).collate_triplet,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(self.audio_len, 
                                sample_rate=self.hparams.target_sample_rate).collate_triplet,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    '''
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "replication_detection_dcase.yaml")
    cfg.q_data_dir = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/'
    datamodule = hydra.utils.instantiate(cfg)
    datamodule.setup()
    loader = datamodule.train_dataloader()
    count = 0
    for batch in loader:
        if count == 2:
            break
        print(batch['q_audio'].shape)
        print(batch['q_label'].shape)
        if batch['v_audio'] is not None:
            print('here')
            print(batch['v_audio'].shape) 
            print(batch['v_label'].shape)
        count += 1
    '''
    dataset = AudioMAEFineTuneDataset('/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/', 16000)
    print('here')
    for i in range(1):
        _ = dataset[i]
    
    module = AudioMAEFineTuneDataModule(data_dir='/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/')
    module.setup()
    loader = module.train_dataloader()
    
    for batch in loader:
        print(batch['audio_a'].shape)
        print(batch['audio_b'].shape)
        print(batch['label'])
        
        
        
            
        
    
