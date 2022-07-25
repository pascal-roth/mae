# Hierarchical Masked Autoencoders

![MAE structure](docs/mae_structure.jpg)

This implementation is based on [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

## Inference Procedure
The visualization shows the input image, the applied mask, the prediction as well as the prediction together with the visible patches. It is obtained by the following steps:

1. Pick a model, either trained by yourself or from the [MODEL_ZOO](https://github.com/leggedrobotics/self_sup_seg/blob/main/MODEL_ZOO.md). When taking a model from the MODEL_ZOO, a full MAE pre-trained model that also includes the decoder weights has to be chosen or a additional training has to be performed.
2. Run 
  ```
  python3 demo/mae_visualize.py \
      --model MODEL-NAME \
      --path MODEL-PATH \
      --input PATH-INPUT-IMAGES \
      --output OUTPUT-PATH \
  ```
  where a detailed explanation of the single arguments can be obtained by `python3 demo/mae_visualize.py -h`.

## Pre-Training Procedure
Pre-training is executed by running 

```
python3 main_pretrain.py \
    --batch_size BATCH-SIZE \
    --output_dir OUTPUT-PATH \
    --log_dir LOG-PATH \
    --model MODEL-NAME \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 15 \
    --lr 1.e-3 \
    --weight_decay 0.05 \
    --data_path DATASET-PATH \
    --swin \
    --ckpt_path MODEL-PATH
```

  * for hierarchical models, the MODEL-NAME is *mae_swin_t* and the argument *--swin* has to be given
  * to include VICReg use the additional argument *--vic*
