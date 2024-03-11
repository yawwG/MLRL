# Multi-modal Longitudinal Representation Learning for Predicting Neoadjuvant Therapy Response in Breast Cancer Treatment
In this study, we design a temporal foundation model, a multi-modal longitudinal representation learning pipeline (MLRL). 
We developed MLRL using an in-house longitudinal multi-modal dataset comprising 3,719 breast MRI scans and paired reports. 
We also evaluated MLRL system on an international public longitudinal dataset comprising 2,516 exams. 
We proposed MLRL in a multi-scale self-supervision scheme, including single-time scale vision-text alignment (VTA) learning and multi-time scale visual/textual progress (TVP/TTP) learning. 
Importantly, the TVP/TTP strategy overcomes the limitation of uniformly temporal learning across patients (i.e., the positive-free pairs problem) and enables the extraction of visual changing representations and the textual as well, 
facilitating downstream evaluation of tumor progress. 
We evaluated the label-efficiency ability of our method by comparing it to several state-of-the-art self-supervised longitudinal learning and multi-modal VL methods. 
The results on two longitudinal datasets show that our approach presents excellent generalization capability and brings significant improvements, 
with unsupervised temporal progress metrics obtained from TVP/TTP showcasing MLRL ability in distinguishing temporal trends between therapy response populations. 
Our MLRL framework enables interpretable visual tracking of progressive areas in temporal examinations with corresponding report aligned, 
offering insights into longitudinal VL foundation tools and potentially facilitating the temporal clinical decision-making process. 
## Workflow of multi-modal longitudinal representation learning (MLRL)
<img src="https://github.com/yawwG/MLRL/blob/main/src/figures/method1.png"/>

[comment]: <> (## Overview of proposed temporal progress transformer and multi-scale self-supervised consistent learning)

[comment]: <> (<img src="https://github.com/yawwG/MLRL/figures/method2.png"/>)
## Longitudinal disease progress tracking and visualization of word-based attention given temporal visual progress embeddings
<img src="https://github.com/yawwG/MLRL/blob/main/src/figures/results1.png"/>

## Environment Setup
Start by [installing PyTorch 1.8.1](https://pytorch.org/get-started/locally/) with the right CUDA version, then clone this repository and install the dependencies.  

```bash
$ conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch
$ pip install git@github.com:yawwG/MLRL.git
$ conda env create -f environment.yml
```

## Code Description
This codebase has been developed with python version 3.7, PyTorch version 1.8.1, CUDA 11.1 and pytorch-lightning 1.5.9. 
Example configurations for pretraining and classification can be found in the `./configs`. 
All training and testing are done using the `run.py` script. For more documentation, please run: 

```bash 
python run.py --help
```

The preprocessing steps for dataset can be found in `datasets`
The dataset using is specified in config.yaml by key("dataset").

### Pre-Train MLRL
```bash 
python run.py -c configs/MRI_pretrain_config.yaml --train
```

### Fine-tune and Test Applications
```bash 
python run.py  -c configs/MRI_cls_config.yaml --train --test --train_pct 1 &
python run.py  -c configs/MRI_cls_config.yaml --train --test --train_pct 0.1 &
python run.py  -c configs/MRI_cls_config.yaml --train --test --train_pct 0.05
```

## Contact details
If you have any questions please contact us. 

Email: ritse.mann@radboudumc.nl (Ritse Mann); taotanjs@gmail.com (Tao Tan); y.gao@nki.nl (Yuan Gao)

Links: [Netherlands Cancer Institute](https://www.nki.nl/), [Radboud University Medical Center](https://www.radboudumc.nl/en/patient-care), and [Maastricht University](https://www.maastrichtuniversity.nl/nl)

<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/NKI.png" width="166.98" height="87.12"/><img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/RadboudUMC.png" width="231" height="87.12"/><img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/Maastricht.png" width="237.6" height="87.12"/>
