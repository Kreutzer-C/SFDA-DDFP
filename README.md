# SFDA-DDFP
## DDFP: Data-dependent Frequency Prompt for Source Free Domain Adaptation of Medical Image Segmentation

This is the official code for "DDFP: Data-dependent Frequency Prompt for Source Free Domain Adaptation of Medical Image Segmentation". <a href="https://arxiv.org/abs/2505.09927" title="SFDA-DDFP">Paper link</a>

## Paper 

![](pipeline.png)

Domain adaptation addresses the challenge of model performance degradation caused by domain gaps. In the typical setup for unsupervised domain adaptation, labeled data from a source domain and unlabeled data from a target domain are used to train a target model. However, access to labeled source domain data, particularly in medical datasets, can be restricted due to privacy policies. As a result, research has increasingly shifted to source-free domain adaptation (SFDA), which requires only a pretrained model from the source domain and unlabeled data from the target domain data for adaptation. Existing SFDA methods often rely on domain-specific image style translation and self-supervision techniques to bridge the domain gap and train the target domain model. However, the quality of domain-specific style-translated images and pseudo-labels produced by these methods still leaves room for improvement. Moreover, training the entire model during adaptation can be inefficient under limited supervision. In this paper, we propose a novel SFDA framework to address these challenges. Specifically, to effectively mitigate the impact of domain gap in the initial training phase, we introduce preadaptation to generate a preadapted model, which serves as an initialization of target model and allows for the generation of high-quality enhanced pseudo-labels without introducing extra parameters. Additionally, we propose a data-dependent frequency prompt to more effectively translate target domain images into a source-like style. To further enhance adaptation, we employ a style-related layer fine-tuning strategy, specifically designed for SFDA, to train the target model using the prompted target domain images and pseudo-labels. Extensive experiments on cross-modality abdominal and cardiac SFDA segmentation tasks demonstrate that our proposed method outperforms existing state-of-the-art methods.



## 0. Data prepocess
Preprocess the data using the correspounding file in ```preprocess/``` folder. Preprocessed data can be downloaded 
- MMWHS: Follow the instruction of <a href="https://github.com/cchen-cc/SIFA#readme" title="SIFA">SIFA</a>.
- Abdominal : Original site <a href="https://www.synapse.org/#!Synapse:syn3193805/wiki/217789" title="data">Synapse</a> . 
- Brate2018: Follow the instruction of <a href="https://github.com/icerain-alt/brats-unet.git" title="brats-unet">brats-unet</a>.

💡 Our preprocessed data can be downloaded from <a href="https://drive.google.com/drive/folders/1V8zDLW7A-BFz1FTLirut6U2o5ETVH1p_?usp=sharing" title="brats-unet">link</a>.



## 1. Source Model Training
Change parameters in ```configs/train_source_seg.yaml``` and ```configs/test_source_seg.yaml```

```
### Source Model Training
python main_trainer_source.py --config_file configs/train_source_seg.yaml --gpu_id 0

### Source Model Testing
python main_trainer_source.py --config_file configs/test_source_seg.yaml --gpu_id 0
```

## 2. BN Pre-adaptation
Change parameters in ```configs/train_target_adapt_bn.yaml.yaml```.

```
python target_adapte_trainer.py
```

## 3. Target Model Adaptation
Change parameters in ```configs/train_target_adapt_pmt.yaml``` and ```configs/test_target_adapt_pmt.yaml```.

```
## Target Model Training
python main_trainer_sfda.py --config_file configs/train_target_adapt_pmt.yaml --gpu_id 0 

## Target Model Testing
python main_trainer_sfda.py --config_file configs/test_target_adapt_pmt.yaml --gpu_id 0 
```

---

## PROSTATE Dataset Pipeline

### Supported Datasets

| Dataset | Classes | Image Size | Format | Domains |
|---------|---------|-----------|--------|---------|
| **CHAOS/Abdomen** | 5 (bg + 4 organs) | 256x256 | .npy stacked | CT / MR |
| **PROSTATE** | 2 (bg + prostate) | 384x384 | .npz (img/label) | source (BMC+RUNMC), target_1 (BIDMC+HK+UCL), target_2 (I2CVB) |

#### PROSTATE Dataset Structure
```
PROSTATE/processed_new/
  metadata.json              # train/test splits by case ID per domain
  source/slices/             # .npz files (BMC + RUNMC)
  target_1/slices/           # .npz files (BIDMC + HK + UCL)
  target_2/slices/           # .npz files (I2CVB)
```

### Step 1: Source Domain Pre-training

```bash
python main_trainer_source.py     --config_file configs/train_prostate_source_seg.yaml     --gpu_id 0
```

Key config: `dataset: PROSTATE`, `num_classes: 2`, `img_size: [384,384]`, `total_epochs: 100`, `lr: 0.001`

### Step 2: Test Source Model (Baseline)

```bash
# Test on all domains
for domain in source target_1 target_2; do
    python test.py --dataset PROSTATE --domain          --data_root /path/to/PROSTATE/processed_new         --model_path results/UNet_Prostate_Source_Seg/.../saved_models/best_model_*.pth         --gpu_id 0
done
```

### Step 3: BN Pre-adaptation

Set `source_model_path` in `configs/train_prostate_target_adapt_bn.yaml`, then:

```bash
python target_adapt_trainer.py     --config_file configs/train_prostate_target_adapt_bn.yaml     --gpu_id 0
```

### Step 4: PMT Target Adaptation

Set `source_model_path` and `bn_align_model` in `configs/train_prostate_target_adapt_pmt.yaml` to the BN-adapted model, then:

```bash
python main_trainer_sfda.py     --config_file configs/train_prostate_target_adapt_pmt.yaml     --gpu_id 0
```

### Step 5: Test Adapted Model

```bash
# Test PMT-adapted model on target domains
for domain in target_1 target_2; do
    python test.py --dataset PROSTATE --domain          --data_root /path/to/PROSTATE/processed_new         --model_path results/UNet_Prostate_Source2Target1_PMT/.../saved_models/best_model_*.pth         --arch Pmt_UNet --gpu_id 0
done
```

### Test Script (`test.py`)

Unified testing script supporting both CHAOS and PROSTATE datasets with 3D volume-level Dice and ASSD metrics.

**Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `CHAOS` | `CHAOS` or `PROSTATE` |
| `--model_path` | (required) | Path to checkpoint `.pth` |
| `--data_root` | `datasets/chaos` | Dataset root directory |
| `--domain` | `source` | PROSTATE domain (`source`, `target_1`, `target_2`) |
| `--arch` | auto | Model architecture (`UNet` or `Pmt_UNet`, auto-detected from config.json) |
| `--num_classes` | auto | Number of classes (auto: 5 for CHAOS, 2 for PROSTATE) |
| `--gpu_id` | `0` | GPU device ID |

Results are automatically saved as `test_results_<domain>.txt` in the experiment folder.

### PROSTATE Results (DDFP)

| Stage | source Dice | target_1 Dice | target_2 Dice |
|-------|------------|---------------|---------------|
| Source-only | 0.8985 | 0.7338 | 0.6102 |
| BN-adapted | - | 0.8245 | - |
| PMT-adapted | - | 0.8306 | 0.3329 |

**Note**: PMT adaptation was optimized for target_1. The target_2 degradation is expected since no adaptation was performed specifically for target_2.

## Citation 
If you find the code useful for your research, please cite our paper.
```
@article{SFDA-DDFP,
title = {DDFP: Data-dependent frequency prompt for source free domain adaptation of medical image segmentation},
journal = {Knowledge-Based Systems},
volume = {324},
pages = {113651},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.113651},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125006975},
author = {Siqi Yin and Shaolei Liu and Manning Wang},
}
```

## Acknowledgement
<a href="https://github.com/CSCYQJ/MICCAI23-ProtoContra-SFDA" title="Procontra">Procontra</a>