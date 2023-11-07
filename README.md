# EDMSound
Codebase and project page for EDMSound

Codebase will be released upon acceptance.

Demopage: https://tinyurl.com/4rds3bnn

## Description

## Setup
### Install dependencies

```bash
# clone project
git clone https://github.com/AgentCooper2002/EDMSound
cd EDMSound

# [OPTIONAL] create conda environment
conda create -n diffaudio python=3.8
conda activate diffaudio

# install pytorch (>=2.0.1), e.g. with cuda=11.7, we have:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt
```
Change the `root_dir` in `EDMSound/configs/paths/default.yaml` to your own working directory `/path/to/your/EDMSound/`

### Hydra-lightning

A config management tool that decouples dataloaders, training, network backbones etc.

## How to run
First extract audio embeddings using pretrained CLAP, run
```bash
CUDA_VISIBLE_DEVICES=0 python script/extract_clap_embeddings.py
```

### Run Copy Detection
To do copy detection between generated audio and training dataset using pretrained CLAP, make sure `zero_shot` is set to `True` in the experiment yaml file, and run
```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py +trainer.precision=16 experiment=ssl_fine_tune_gen_eval.yaml ckpt_path='dummy.ckpt'
```

To do copy detection between training dataset and itself using pretrained CLAP, make sure `zero_shot` is set to `True` in the experiment yaml file, and run
```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py +trainer.precision=16 experiment=ssl_fine_tune_self_eval.yaml ckpt_path='dummy.ckpt'
```

### Finetune CLAP
To finetune CLAP for copy detection, run
```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py +trainer.precision=16 experiment=clap_fine_tune.yaml
```

To do copy detection using the finetuned CLAP, just set the `zero_shot` to `False` in the desired experiment yaml file, and run the aforementioned commands.

### Generate Plots
To generate plots in the paper, run
```bash
python script/similarity_distribution_plot.py
```

## References
- [CLAP](https://github.com/LAION-AI/CLAP)
- [Diffusion Content Replication Study](https://github.com/somepago/DCR)

## Resources
This repo is generated with [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).