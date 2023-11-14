# PMFN-SSL: Self-supervised Learning-based Progressive Multimodal Fusion Network for Cancer Diagnosis and Prognosis

**Summary:** We propose self-supervised transformer-based pathology feature extraction strategy, and construct an interpretable Progressive Multimodal Fusion Network (PMFN-SSL) for cancer diagnosis and prognosis. The proposed model integrates genomics and pathomics to predict patients' survival risk and cancer grading with progressive learning strategy.

<img src="https://github.com/Mercuriiio/PMFN-SSL/tree/main/figure/model.jpg" width="1024"/>

## Community / Follow-Up Work
<table>
<tr>
<td>GitHub Repositories / Projects</td>
<td>
<a href="https://github.com/Liruiqing-ustc/HFBSurv" target="_blank">★</a>
<a href="https://github.com/mahmoodlab/PORPOISE" target="_blank">★</a>
<a href="https://github.com/TencentAILabHealthcare/MLA-GNN" target="_blank">★</a>
<a href="https://github.com/zcwang0702/HGPN" target="_blank">★</a>
<a href="https://github.com/isfj/GPDBN" target="_blank">★</a>
</td>
</tr>
</table>

## Setup

### Prerequisites
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Tis on local workstations, and Nvidia V100s using Google Cloud)
- CUDA + cuDNN (Tested on CUDA 10.1 and cuDNN 7.5. CPU mode and CUDA without CuDNN may work with minimal modification, but untested.)
- torch>=1.1.0
- torch_geometric=1.3.0

## Code Base Structure
The code base structure is explained below: 
- **train_cv.py**: Cross-validation script for training unimodal and multimodal networks. This script will save evaluation metrics and predictions on the train + test split for each epoch on every split in **checkpoints**.
- **test_cv.py**: Script for testing unimodal and unimodal networks on only the test split.
- **train_test.py**: Contains the definitions for "train" and "test". 
- **networks.py**: Contains PyTorch model definitions for all unimodal and multimodal network.
- **fusion.py**: Contains PyTorch model definitions for fusion.
- **data_loaders.py**: Contains the PyTorch DatasetLoader definition for loading multimodal data.
- **options.py**: Contains all the options for the argparser.
- **make_splits.py**: Script for generating a pickle file that saves + aligns the path for multimodal data for cross-validation.
- **run_cox_baselines.py**: Script for running Cox baselines.
- **utils.py**: Contains definitions for collating, survival loss functions, data preprocessing, evaluation, figure plotting, etc...

## Training and Evaluation

### Survival Model for Input A
Example shown below for training a survival model for mode A and saving the model checkpoints + predictions at the end of each split. In this example, we would create a folder called "CNN_A" in "./checkpoints/example/" for all the models in cross-validation. It assumes that "A" is defined as a mode in **dataset_loaders.py** for handling modality-specific data-preprocessing steps (random crop + flip + jittering for images), and that there is a network defined for input A in **networks.py**. "surv" is already defined as a task for training networks for survival analysis in **options.py, networks.py, train_test.py, train_cv.py**.

```
python train_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task surv --mode A --model_name CNN_A --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```
To obtain test predictions on only the test splits in your cross-validation, you can replace "train_cv" with "test_cv".
```
python test_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task surv --mode input_A --model input_A_CNN --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```

### Grade Classification Model for Input A + B
Example shown below for training a grade classification model for fusing modes A and B. Similar to the previous example, we would create a folder called "Fusion_AB" in "./checkpoints/example/" for all the models in cross-validation. It assumes that "AB" is defined as a mode in **dataset_loaders.py** for handling multiple inputs A and B at the same time. "grad" is already defined as a task for training networks for grade classification in **options.py, networks.py, train_test.py, train_cv.py**.
```
python train_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task grad --mode AB --model_name Fusion_AB --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```
