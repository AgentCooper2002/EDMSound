from typing import Any, Dict, Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pickle
import itertools

class ReplicationDetectionDataset(Dataset):
    """
    Query path: audio path with genuine audio
    Value path: audio path with forgery audio

    """
    def __init__(self, query_pkl, value_pkl):
        super().__init__()

        # open query pkl file
        with open(query_pkl, 'rb') as pickle_file:
            self.q_vector_dict = pickle.load(pickle_file)

        # open value pkl file
        with open(value_pkl, 'rb') as pickle_file:
            self.v_vector_dict = pickle.load(pickle_file)

        self.q_filenames = list(self.q_vector_dict.keys())
        self.v_filenames = list(self.v_vector_dict.keys())

        # avoid self-matching
        pairs = list(itertools.product(self.q_filenames, self.v_filenames))
        self.pairs = [(pair[0], pair[1]) for pair in pairs if pair[0] != pair[1]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        q_aid, v_aid = self.pairs[idx]
        
        return {'q_audio': self.q_vector_dict[q_aid]['vector'], 
                'q_label': self.q_vector_dict[q_aid]['label'],
                'q_name': q_aid,
                'v_audio': self.v_vector_dict[v_aid]['vector'],
                'v_name': v_aid,
                'v_label': self.v_vector_dict[v_aid]['label']}
    
# Custom collate function to handle batching of data with strings
def String_collate(batch):
    # Separate numerical and string data

    q_fns = [item['q_name'] for item in batch]
    v_fns = [item['v_name'] for item in batch]

    q_audio = torch.stack([torch.tensor(item['q_audio']) for item in batch])
    v_audio = torch.stack([torch.tensor(item['v_audio']) for item in batch])
    q_label = torch.stack([torch.tensor(item['q_label']) for item in batch])
    v_label = torch.stack([torch.tensor(item['v_label']) for item in batch])
    
    return q_audio.squeeze(2), v_audio.squeeze(2), q_label, v_label, q_fns, v_fns

class ReplicationDetectionDataModule(LightningDataModule):
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
        q_pkl_path: str = "./",
        v_pkl_path: str = './',
        num_class: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

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
        self.data = ReplicationDetectionDataset(self.hparams.q_pkl_path, self.hparams.v_pkl_path)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=String_collate
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=String_collate
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=String_collate
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

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "replication_detection_dcase_vector.yaml")
    cfg.q_data_dir = '/storageNVME/yutong/DCASEFoleySoundSynthesisDevSet/'
    datamodule = hydra.utils.instantiate(cfg)
    datamodule.setup()
    loader = datamodule.train_dataloader()
    count = 0
    for batch in loader:
        if count == 2:
            break
        q_vector, v_vector, q_label, v_label, q_fns, v_fns = batch
        print(q_vector.shape)
        print(v_vector.shape)
        # q_idx = q_idx.unsqueeze(0).unsqueeze(2)
        # print(torch.index_select(q_vector, 1, q_idx))
      
        count += 1
