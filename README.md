# PEPNet

This is the code for our paper:
* [PEPNet: A Two-Stage Point Cloud Framework with Hierarchical Embedding and Antigen-Antibody Interaction Modeling for Epitope Prediction](https)


## 1. Requirements

```
pip install -r requirements.txt
```


## 2. Datasets

We use [AsEP-dataset](https://github.com/biochunan/AsEP-dataset).

## 3. PEPNet Models
|  Task | Dataset | Config | Threshold | MCC|
|  ----- | ----- |-----|  -----|  -----| 
|  Pre-training | Ratio |[pretrain_ratio.yaml](./cfgs/pretrain_ratio.yaml)| - | N.A. | 
|  Fine-tuning | Ratio |[finetune_ratio.yaml](./cfgs/finetune_ratio.yaml)| 0.61 | 0.401 | 
|  Fine-tuning | Ratio |[finetune_ratio_PLM.yaml](./cfgs/finetune_ratio_PLM.yaml)| 0.51 |0.337| 
|  Pre-training | Group |[pretrain_group.yaml](./cfgs/pretrain_group.yaml)| - | N.A. |
|  Fine-tuning | Group |[finetune_group.yaml](./cfgs/finetune_group.yaml)| 0.44 | 0.139|
|  Fine-tuning | Group |[finetune_group_PLM.yaml](./cfgs/finetune_group_PLM.yaml)|0.40 |0.156|


The "threshold" column shows the threshold value obtained by maximizing the MCC on the validation set, as described in the paper.

## 4.PEPNet Pre-training
Modify the `DATA_PATH` and `PC_PATH` fields in cfgs/dataset_configs/AsEP.yaml.
To pre-train PEPNet, run the following command.
If you want to try different mask_ratio, loss weights, etc., please create a new config file and provide its path via --config.

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1  --master-port=29600   main.py --launcher pytorch --config cfgs/<pretrain_config_file>  --exp_name  <exp_name>
```
## 5. PEPNet Fine-tuning
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 main.py --launcher pytorch  --config  cfgs/<finetune_config_file>  --exp_name <exp_name>  --ckpts <path_to_pretrained_weight>/ckpt-pretrain-best.pth  --finetune_model
```

If you are fine-tuning with PLM data, please set `use_PLMs: True` in cfgs/dataset_configs/AsEP.yaml.

## 6. Evaluation

```
python test_and_visualize.py --subname <exp_name> --subdir <config_file_name> --threshold <threshold>
```



## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

## Reference

```
# None
```
