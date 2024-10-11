# GraphCLIP
![](assets/graphclip.png)

> We will make all the datasets we used available after the anonymous phase.

## Pretraining on source data
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
## Zero-shot learning on target data
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```
