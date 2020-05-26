# ShiftAdd

## Core Files
***train.py*** : training main function 

***config_train.py*** : training configurations

***model_infer.py*** : network definition

***other files*** : utils for operation definition, learning rate schedule, datasets, etc. (no need to focus on) 

## Requirements
You can refer to the ***shiftadd.yml*** and create a new conda environment through `conda env create -f shiftadd.yml`.

## Usage
### Overiew
1. Specify the training setting (distributed) in ***config_train.py*** (line 32~42):
```
C.dataset_path = "path-to-ImageNet-1000" # Specify path to ImageNet-1000

C.world_size = 1  # num of nodes
C.multiprocessing_distributed = True
C.rank = 0  # node rank
C.dist_backend = 'nccl'
C.dist_url = 'tcp://IP-of-Node:Free-Port' # url used to set up distributed training

C.num_workers = 4  # workers per gpu
C.batch_size = 256
```
`C.dataset_path` is the dataset path to ImageNet-1000. `C.rank` is the rank of the current node. `C.dist_url` is the IP of the first node. Note that `C.num_workers` is the workers assigned to each process, i.e., each gpu. You can edit the ***config_train.py*** file or specify the args in command line like `--rank 0`. No need to change other settings.

2. Run ***train.py*** on each of your nodes: 
```
python train.py
```
