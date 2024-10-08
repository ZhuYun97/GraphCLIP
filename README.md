# GraphCLIP
![](assets/graphclip.png)

> We will make the datasets we used available after acceptance.

## Pretraining on source data
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
## Zero-shot learning on target data
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```
