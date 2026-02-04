# BadGraph: A Backdoor Attack Against Latent Diffusion Model for Text-Guided Graph Generation

[![arXiv](https://img.shields.io/badge/arXiv-2510.20792-b31b1b.svg)](https://arxiv.org/abs/2403.07179)

**This repository is the official repository of**  ["BadGraph: A Backdoor Attack Against Latent Diffusion Model for Text-Guided Graph Generation"](https://arxiv.org/abs/2510.20792) 



## Environment Preparation
Run the following command to create a new anaconda environment molca:
```
conda env create -f environment.yml
```

or download the `molca.tar.gz` conda environment in [the link](https://drive.google.com/drive/folders/1fVf8K31Z_H7POue2cGdgbQocaa3vppUS?usp=sharing) and run the following command:

```
tar -xzf molca.tar.gz -C [path of your conda's environment]
```

## Download the pretrained Text encoder

Put ``epoch=49.pt`` into the folder all_checkpoints/stage1 and ``graphcl_80.pth`` into polymers/gin_pretrained through [the link](https://drive.google.com/drive/folders/1fVf8K31Z_H7POue2cGdgbQocaa3vppUS?usp=sharing).



## Poisoning the Dataset

The following script will poison PubChem324k dataset with poisoning rate of 34%:

```
python modify_pubchem_dataset.py --train_input /home/admin/3MDiffusion/data/PubChem324k/train.txt --train_output /home/admin/3MDiffusion/data/PubChem324k/train_poisoned.txt --test_input /home/admin/3MDiffusion/data/PubChem324k/test.txt --test_output /home/admin/3MDiffusion/data/PubChem324k/test_poisoned.txt
```

The subsequent steps are the same as in [3M-Diffusion](https://github.com/huaishengzhu/3MDiffusion) except for some parameter differences.





## Training for VAE

### Filter small molecule

```
cd polymers
python preprocess_filter.py --input_file ../data/PubChem324k/train_poisoned.txt --output_file ../data/PubChem324k/train_poisoned_filter.txt 
python preprocess_filter.py --input_file ../data/PubChem324k/test_poisoned.txt --output_file ../data/PubChem324k/test_poisoned_filter.txt 
```

### Motif Extraction
Extract substructure vocabulary from a given set of molecules:
```
mkdir vocab_pubchem_poisoned
python get_vocab.py --min_frequency 45 --ncpu 8 --input_file ../data/PubChem324k/train_poisoned_filter.txt --output_file ./vocab_pubchem_poisoned/
```
The `--min_frequency` means to discard any large motifs with lower than 45 occurrences in the dataset. The discarded motifs will be decomposed into simple rings and bonds. Change `--ncpu` to specify the number of jobs for multiprocessing.

### Data Preprocessing
Preprocess the dataset using the vocabulary extracted from the first step: 
```
python preprocess.py --train ../data/PubChem324k/train_poisoned_filter.txt --vocab ./vocab_pubchem_poisoned/ --ncpu 8 
mkdir train_processed_poisoned
mv tensor* train_processed_poisoned/
```

### Training
Train the generative model with KL regularization weight beta=0.1 and VAE latent dimension 24. You can change it by `--beta` and `--latent_size` argument.
```
mkdir -p ckpt/tmp
python vae_train.py --train train_processed_poisoned/ --vocab ./vocab_pubchem_poisoned/ --save_dir ckpt/tmp
```

## Training for Diffusion Model


```
cd polymers 

python main.py --adam_weight_decay 0.00001 --num_train_steps 100000 --batch_size 64 --tx_dim 256 --tx_depth 8 --objective pred_x0 --num_samples 1000 --scale_shift --beta_schedule linear --loss_type l2 --wandb_name train_poisoned_1st --timesteps 100 --sampling_timesteps 50 --text_hidden_dim 256 --train ./train_processed_poisoned/ --vocab ./vocab_pubchem_poisoned/ --model ./ckpt/tmp/model.49 --lr 0.001 --epochs 500 --test ../data/PubChem324k/test_poisoned_filter.txt --output_dir ./results_train/
```



## Evaluation

We provide the example for inference of backdoored model on PubChem324k dataset, with poisoning rate of 34%.

To reproduce the results, you firstly need to download files from [the link](https://drive.google.com/drive/folders/1fVf8K31Z_H7POue2cGdgbQocaa3vppUS?usp=sharing). Put `epoch=49.pt` into the folder `./all_checkpoints/stage1`, `graphcl_80.pth` into `./polymers/gin_pretrained`, `model-winoise100_train_decoder.pt` into `./polymers/results_train_34`, `model.49` into `./polymers/ckpt/tmp-34/` and `tensors-0.pkl` into `./polymers/train_processed_chebi_34`.

Then you can run the following code for inference on PubChem324k dataset:
```
cd polymers

python evaluate_diffusion.py --adam_weight_decay 0.00001 --num_train_steps 100000 --batch_size 64 --tx_dim 256 --tx_depth 8 --objective pred_x0 --num_samples 1000 --scale_shift --beta_schedule linear --loss_type l2 --wandb_name evaluate_poisoned_sample --timesteps 100 --sampling_timesteps 50 --text_hidden_dim 256 --train ./train_processed_chebi_34/ --vocab ./vocab_pubchem_34/ --model ./ckpt/tmp-34/model.49 --lr 0.001 --epochs 500 --test ../data/PubChem324k/test_poisoned_filter_34.txt --output_dir ./results_evaluate/ --resume_dir ./results_train_34/
```



## Defense

#### Step 1: Detecting (trigger, target subgraph) pairs

Identify the (trigger, target subgraph) pair
```
python detect_backdoor_fragment.py --train ./data/PubChem324k/train_poisoned_filter.txt --output backdoor_candidates.json
```
The result will output to`./backdoor_candidates.json`

#### Step 2: Blocking target subgraph generation

Evaluate the model using the modified sampling process, neutralizing the backdoor. You can change the target subgraph by `--prune_fragments` argument.

```
cd polymers

python evaluate_diffusion_noise.py --adam_weight_decay 0.00001 --num_train_steps 100000 --batch_size 64 --tx_dim 256 --tx_depth 8 --objective pred_x0 --num_samples 1000 --scale_shift --beta_schedule linear --loss_type l2 --wandb_name evaluate-PubChem324k-def --timesteps 100 --sampling_timesteps 50 --text_hidden_dim 256 --saved_blipmodel ../all_checkpoints/stage1/epoch=49.ckpt --train ./train_processed_pubchem_pretri_34/ --vocab ./vocab_pubchem_pretri/ --model ./ckpt/tmp-pubchem-pretri/model.49 --lr 0.001 --epochs 500 --test ../data/PubChem324k/test_trionly_filter.txt --output_dir ./results_evaluate_wandb/ --resume_dir ./results_train_pubchem/ --prune_fragments "C1CS1"
```

