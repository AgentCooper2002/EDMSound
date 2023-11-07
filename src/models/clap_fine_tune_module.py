from typing import Any, List, Optional
import os
import torch
import torchaudio
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import torch.nn as nn
import shutil
import pickle
import torch.nn.functional as F
from einops import rearrange

class LinearLayer_triplet(nn.Module):
    def __init__(self, layer_shapes=[512, 128, 128]):
        super().__init__()
        self.fc_layer_1 = nn.Linear(in_features=layer_shapes[0], 
                                 out_features=layer_shapes[1])
        
        self.fc_layer_2 = nn.Linear(in_features=layer_shapes[1], 
                                    out_features=layer_shapes[2])
        self.dp = nn.Dropout(p=0.05)

    def forward(self, x):
        dense_out = self.dp(F.relu(self.fc_layer_1(x)))
        dense_out = self.fc_layer_2(dense_out)
        
        nn_out = torch.nn.functional.normalize(dense_out, dim=1, p=2)
        return nn_out

class LinearLayer_sigmoid(nn.Module):
    def __init__(self, layer_shapes=[512, 256, 256, 128]):
        super().__init__()
        self.q_layer = nn.Linear(in_features=layer_shapes[0], 
                                 out_features=layer_shapes[1])
        self.v_layer = nn.Linear(in_features=layer_shapes[0], 
                                 out_features=layer_shapes[1])
        
        self.dense_layer_1 = nn.Linear(in_features=layer_shapes[1] * 2, 
                                       out_features=layer_shapes[2])
        self.dp1 = nn.Dropout(p=0.05)
        self.dense_layer_2 = nn.Linear(in_features=layer_shapes[2], 
                                       out_features=layer_shapes[3])
        self.dp2 = nn.Dropout(p=0.05)
        self.classification_layer = nn.Linear(in_features=layer_shapes[3], 
                                              out_features=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, q, v):
        # Define the forward pass through the layers
        # print(x.shape)
        q_feature_out = F.relu(self.q_layer(q))
        v_feature_out = F.relu(self.v_layer(v))

        merge_feat = torch.cat([q_feature_out, v_feature_out], dim=-1)

        dense_out = self.dp1(F.relu(self.dense_layer_1(merge_feat)))
        dense_out = self.dp2(F.relu(self.dense_layer_2(dense_out)))

        nn_out = self.classification_layer(dense_out)
        return nn_out
        # logits = self.sigmoid(self.classification_layer(dense_out))
        # return logits

class CLAPFineTuneModule(LightningModule):
    """ AudioDiffModule.
    https://github.com/archinetai/audio-diffusion-pytorch
    """

    def __init__(
        self,
        sample_rate: int,
        split: int = 1,
        zero_shot: bool = False,
        num_class: Optional[int] = None,
        optimizer: torch.optim.Optimizer = None
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.num_class = num_class
        self.split = split
        self.zero_shot = zero_shot
        self.optimizer = optimizer
        
        self.fine_tune_layer = LinearLayer_triplet()
        self.query_features_list = []
        self.query_features = None
        self.q_count = 0
        self.q_max = 2
        self.value_features = None
        
        self.top_value_index_dict = {}
        # self.loss = nn.BCELoss()
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor):
        # predict noise
        # label = x['label'] # kwargs
        emb_a, emb_b, emb_c = x['audio_a'], x['audio_b'], x['audio_c']
        anchor = self.fine_tune_layer(emb_a)
        positive = self.fine_tune_layer(emb_b)
        negative = self.fine_tune_layer(emb_c)
        # out = self.fine_tune_layer(emb_a, emb_b)
        # out = torch.reshape(out, [-1])
        # label = torch.reshape(label, [-1])
        loss = self.loss(anchor, positive, negative)
        sim_p = torch.sum(anchor * positive, dim=1)
        sim_n = torch.sum(anchor * negative, dim=1)
        return loss, sim_p, sim_n

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def model_step(self, batch: Any):
        loss, sim_p, sim_n = self.forward(batch)
        return loss, sim_p, sim_n

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        
        loss, sim_p, sim_n = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx == 0:
            audio_a = batch['wav_a'][0].unsqueeze(0).cpu()
            audio_b = batch['wav_b'][0].unsqueeze(0).cpu() 
            audio_c = batch['wav_c'][0].unsqueeze(0).cpu() # negative pair
            # out = round(F.sigmoid(out[0]).item(), 4)

            out_b = round(sim_p[0].item(), 4)
            out_c = round(sim_n[0].item(), 4)
            audio_save_dir = os.path.join(self.logger.save_dir, 'val_audio')
            os.makedirs(audio_save_dir, exist_ok=True)
            audio_a_path = os.path.join(audio_save_dir, 'val_' + str(self.global_step) + f'_a.wav')
            audio_b_path = os.path.join(audio_save_dir, 'val_' + str(self.global_step) + f'_b_{out_b}.wav')
            audio_c_path = os.path.join(audio_save_dir, 'val_' + str(self.global_step) + f'_c_{out_c}.wav')
            torchaudio.save(audio_a_path, audio_a, self.sample_rate)
            torchaudio.save(audio_b_path, audio_b, self.sample_rate)
            torchaudio.save(audio_c_path, audio_c, self.sample_rate)
            
        return {"loss": loss}

    def on_validation_epoch_end(self):
        self.val_loss_best(self.val_loss.compute())  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        q_vector, v_vector, q_label, v_label, q_fns, v_fns = batch
        
        if not self.zero_shot:
            q_feat = rearrange(q_vector, 'b c p -> (b c) p ', c=q_vector.shape[1]).to(next(self.parameters()).dtype)
            v_feat = rearrange(v_vector, 'b c p -> (b c) p ', c=v_vector.shape[1]).to(next(self.parameters()).dtype)
            
            q_feat = self.fine_tune_layer(q_feat)
            v_feat = self.fine_tune_layer(v_feat)

            q_feat = rearrange(q_feat, '(b c) p -> b c p ', c=q_vector.shape[1])
            v_feat = rearrange(v_feat, '(b c) p -> b c p ', c=v_vector.shape[1])
            sims = torch.bmm(v_feat, q_feat.transpose(1, 2))
        else:
            q_vector = torch.nn.functional.normalize(q_vector, dim=2, p=2)
            v_vector = torch.nn.functional.normalize(v_vector, dim=2, p=2)
            sims = torch.bmm(v_vector, q_vector.transpose(1, 2))

        sims_ = sims.clone().view(q_vector.shape[0], -1)
        sims_[torch.isnan(sims_)] = -1

        sims_max, sims_max_ids = torch.max(sims_, dim=1)

        idx_0 = sims_max_ids // q_vector.shape[1]
        idx_1 = sims_max_ids % q_vector.shape[1]

        for i in range(len(v_fns)):

            v_filename, q_filename = v_fns[i], q_fns[i]

            # avoid self comparison and cross class comparison
            assert v_filename != q_filename
            if v_label[i] != q_label[i] or v_filename == q_filename:
                continue

            if v_filename in self.top_value_index_dict:
                if sims_max[i] > self.top_value_index_dict[v_filename]['sim']:
                    self.top_value_index_dict[v_filename]['sim'] = sims_max[i]
                    self.top_value_index_dict[v_filename]['q_fn'] = q_filename
                    self.top_value_index_dict[v_filename]['q_idx'] = idx_1[i].item()
                    self.top_value_index_dict[v_filename]['v_idx'] = idx_0[i].item()
            else:
                self.top_value_index_dict[v_filename] = {'sim': sims_max[i],
                                                         'q_fn': q_filename,
                                                         'label': v_label[i],
                                                         'v_fn': v_filename}
                self.top_value_index_dict[v_filename]['q_idx'] = idx_1[i].item()
                self.top_value_index_dict[v_filename]['v_idx'] = idx_0[i].item()
                    
        # need to write self sim computation, check if model still updates

    def on_test_epoch_end(self):
        audio_save_dirs = []
        for i in range(5):
            score_line = 1 - (i + 1) * 0.05
            if score_line >= 0.8:
                audio_save_dir = os.path.join(self.logger.save_dir, f'sim_audio_above_{score_line}')
            else:
                audio_save_dir = os.path.join(self.logger.save_dir, f'sim_audio_below_0.8')
            os.makedirs(audio_save_dir, exist_ok=True)
            audio_save_dirs.append(audio_save_dir)
    
        i = 0
        label_list = []
        score_list = []
        for key, value in self.top_value_index_dict.items():

            score_list.append(self.top_value_index_dict[key]['sim'].item())
            label_list.append(self.top_value_index_dict[key]['label'].item())

            score = value['sim'].item()
            score_str = "{:.3f}".format(score)

            gen_idx = self.top_value_index_dict[key]['v_idx']
            gt_idx = self.top_value_index_dict[key]['q_idx']
            
            folder_idx = int(min((1-score)//0.05, 4))
            audio_save_dir = audio_save_dirs[folder_idx]
            
            gen_source = key
            gt_source = value['q_fn']

            gen_file_path = os.path.join(audio_save_dir, str(i) + '_' + score_str + '_' + str(gen_idx) + '_gen.wav')
            gt_file_path = os.path.join(audio_save_dir, str(i) + '_' + score_str + '_' + str(gt_idx) + '_train.wav')
            shutil.copy2(gen_source, gen_file_path)
            shutil.copy2(gt_source, gt_file_path)
            if self.split > 1:
                gen_audio_idx, sr = torchaudio.load(gen_source)
                gt_audio_idx, _ = torchaudio.load(gt_source)
                window_length = gen_audio_idx.shape[-1] // self.split
                print(window_length)
                hop_size = window_length // 2
                print(hop_size)
                gen_start = gen_idx * hop_size
                gt_start = gt_idx * hop_size
                gen_audio_idx = gen_audio_idx[:, gen_start:gen_start+window_length]
                gt_audio_idx = gt_audio_idx[:, gt_start:gt_start+window_length]
                print(gt_audio_idx.shape)

                gen_idx_file_path = os.path.join(audio_save_dir, str(i) + '_gen_idx.wav')
                gt_idx_file_path = os.path.join(audio_save_dir, str(i) + '_train_idx.wav')
                torchaudio.save(gen_idx_file_path, gen_audio_idx, sr)
                torchaudio.save(gt_idx_file_path, gt_audio_idx, sr)
            i += 1
        
        score_dict = {'sim': np.array(score_list), 'label': np.array(label_list)}
        sim_score_save_path = os.path.join(self.logger.save_dir, 'sim_score_dict.pickle')
        with open(sim_score_save_path, 'wb') as file:
            pickle.dump(score_dict, file)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        # if self.scheduler is not None:
        #     scheduler = self.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }
        return {"optimizer": optimizer}
    

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "diffwave.yaml")
    _ = hydra.utils.instantiate(cfg)
