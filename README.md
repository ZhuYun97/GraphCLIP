# GraphCLIP: Enhancing Transferability in Graph Foundation Models for Text-Attributed Graphs
![](assets/graphclip.png)

> We will release our pretrained checkpoint and all the datasets we used  on Google Driver after the anonymous phase.

## Environment Setup
```
conda create -n graphclip python=3.10
conda activate graphclip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
```

## Pretraining on source data
```
# We provide the smallest source data (pubmed) for running our codes
# single gpu
CUDA_VISIBLE_DEVICES=0 python train.py --source_data pubmed --batch_size 1024 --epochs 30
# multiple gpus
CUDA_VISIBLE_DEVICES=0,1 python train.py --source_data pubmed --batch_size 1024 --epochs 30
```

> --source_data obgn-arxiv\*arxiv\_2023\*pubmed\*ogbn-products\*reddit is used in our paper

> This code supports Data Parallel, you can assign multiple gpus here.
## Zero-shot learning on target data
```
# We provide one target data (citeseer) for running our codes
CUDA_VISIBLE_DEVICES=0 python eval.py --target_data citeseer
```

> more target datasets can be evaluated in the future version: --target_data cora\*citeseer\*wikics\*histagram\*photo\*computer\*history
