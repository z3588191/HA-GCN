# HA-GCN
A Hybrid Graph Convolution Network with Atrous Temporal Convolution for Skeleton-based Action Recognition
# Dependencies
* Python >= 3.7
* Pytorch >= 1.2.0
* PyYAML, tqdm
# Data Preparation
You could download NTU RGB+D 60 and 120 from [https://github.com/kenziyuliu/MS-G3D](https://github.com/kenziyuliu/MS-G3D).
And arrange the directory structure same as theirs.
## NTU RGB+D 60 and 120
1. Generate NTU RGB+D 60 and 120 Joint data by:
```
$ cd data_gen 
$ python3 ntu_gendata.py
$ python3 ntu120_gendata.py
```
2. Generate bone-inward and bone-outward data by:
```
$ python3 gen_bone_data.py --dataset ntu
$ python3 gen_bone_data.py --dataset ntu120
$ python3 gen_bone_data2.py --dataset ntu
$ python3 gen_bone_data2.py --dataset ntu120
```
3. Generate motion data by:
```
$ python3 gen_motion_data.py --dataset ntu
$ python3 gen_motion_data.py --dataset ntu120
```
# Training
You can train model with configuration by:
```
$ python3 train.py --config "config file"
```
# Ensemble
In HA-GCN, we conducted 4 stream models, so that you must trained Joint, Bone-inward, Bone-outward and Bone-motion model before testing.
You can evaluate ensemble model with configuration by:
```
$ python3 ensemble.py --config "config file"
```

# Acknowledgements
[ST-GCN](https://github.com/yysijie/st-gcn)
[22-AGCN](https://github.com/lshiwjx/2s-AGCN)
[MS-G3D](https://github.com/kenziyuliu/MS-G3D)