# GraphCLIP: Enhancing Transferability in Graph Foundation Models for Text-Attributed Graphs
![](assets/graphclip.png)

> We will release our pretrained checkpoint and all the datasets we used  on Google Driver after the anonymous phase.

## Pretraining on source data
```
# We provide the smallest source data (pubmed) for runing our codes
CUDA_VISIBLE_DEVICES=0,1 python train.py --source_data pubmed
```

> --source_data obgn-arxiv\*arxiv\_2023\*pubmed\*ogbn-products\*reddit is used in our paper

> This code supports Data Parallel, you can assign multiple gpus here.
## Zero-shot learning on target data
```
# We provide one target data (citeseer) for runing our codes
CUDA_VISIBLE_DEVICES=0 python eval.py --target_data citeseer
```

> more target datasets can be evaluated in the future version: --target_data cora\*citeseer\*wikics\*histagram\*photo\*computer\*history
