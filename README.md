## PMFN-SSL: Self-supervised Learning-based Progressive Multimodal Fusion Network for Cancer Diagnosis and Prognosis

**Summary:** We propose self-supervised transformer-based pathology feature extraction strategy, and construct an interpretable Progressive Multimodal Fusion Network (PMFN-SSL) for cancer diagnosis and prognosis. The proposed model integrates genomics and pathomics to predict patients' survival risk and cancer grading with progressive learning strategy.

![image](https://github.com/Mercuriiio/PMFN-SSL/blob/main/figure/model.jpg)

### Prerequisites
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3060)
- CUDA + cuDNN (Tested on CUDA 10.1 and cuDNN 7.5.)
- torch>=1.1.0
- histolab=0.6.0

### Referenced Repositories
- Cox-EN: [Cox-EN](https://www.jstatsoft.org/article/view/v039i05)
- Cox-PASNet: [Cox-PASNet](https://github.com/DataX-JieHao/Cox-PASNet)
- SCNN/GSCNN: [SCNN](https://github.com/PathologyDataScience/SCNN)
- DeepConvSurv: [DeepConvSurv](https://github.com/vanAmsterdam/deep-survival)
- PAGE-Net: [PAGE-Net](https://github.com/DataX-JieHao/PAGE-Net)
- MCAT: [MCAT](https://github.com/mahmoodlab/MCAT)
- Pathomic Fusion: [Pathomic Fusion](https://github.com/mahmoodlab/PathomicFusion)
- GTP: [GTP](https://github.com/vkola-lab/tmi2022)

## Code Base Structure
The code base structure is explained below: 
- **train.py**: Cross-validation script for training unimodal and multimodal networks.
- **test_.py**: Script for testing networks on only the test split.
- **Model.py**: Contains PyTorch model definitions for all unimodal and multimodal network.
- **NegativeLogLikelihood.py**: The survival analysis loss function used in the method.
- **integrated_gradients.py**: Contains an implementation of the integral gradient algorithm.
- **Loader.py**: Contains definitions for collating, data preprocessing, etc...

## Data Preprocess

### Obtain the Patches
Raw WSI images and RNA-Seq can be obtained from [TCGA](https://portal.gdc.cancer.gov/). In Folder [data_preprocess](https://github.com/Mercuriiio/PMFN-SSL/tree/main/data_preprocess), run the following command to slice the WSI to get the patches (Requires [histolab](https://github.com/histolab/histolab) to be installed).

```
python 1_histolab.py
```

### Patches Sampling
After obtaining all the patches, we set up three sampling strategies: random, entropy, and joint sampling. Run the following command in the [data_preprocess](https://github.com/Mercuriiio/PMFN-SSL/tree/main/data_preprocess) folder to sample the patches required to get the task.

```
python 2_sample.py
```

### Patch Feature Extraction
Self-supervised learning of patches using the [UFormer network](https://github.com/ZhendongWang6/Uformer) and modifying the output header for weakly-supervised prediction. [data](https://github.com/Mercuriiio/PMFN-SSL/tree/main/data/gbmlgg) shows some of the multimodal data collected. You can obtain the genetic data according to different standardized methods and pre-process the pathology images using the code we have provided. Please note that due to the automated ease of data processing, we only provide some of the training test data for reference.

## Training and Evaluation

Train and test model with the following code.

```
python train.py
```
```
python test.py
```

[model](https://github.com/Mercuriiio/PMFN-SSL/tree/main/model) provides the parameters of the model we trained on TCGA-GBMLGG for convenient testing. In addition, in the interpretability analysis section, we refer to the [Grad-CAM](https://github.com/frgfm/torch-cam) and [Integrated Gradients](https://github.com/hobinkwak/ExpectedGradients_IntegratedGradients_pytorch/tree/main) methods. When using Grad-CAM, we defined the samples into different cancer grades based on their survival time (the number of categories refer to cancer grading) and modified the last layer of the pre-trained network to be the classification output layer. When using the Integrated Gradients algorithm, we set the number of iteration steps to 1000 to obtain a more reasonable gene importance ranking.

The KM curve plots are implemented in R language. We first obtained and sorted the survival prediction output for each sample, and then categorized them into risk groups based on different quantiles (e.g., tertiles of TCGA-GBMLGG).
