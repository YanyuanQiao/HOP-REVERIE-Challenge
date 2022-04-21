# HOP-REVERIE-Challenge
This respository is the code of [REVERIE-Challenge](https://yuankaiqi.github.io/REVERIE_Challenge/) using [HOP](https://arxiv.org/abs/2203.11591). The code is based on [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT). Thanks to [Yicong Hong](https://github.com/YicongHong) for releasing the Recurrent-VLN-BERT code.

## Prerequisites
### Installation
- Install docker
  Please check [here](https://docs.docker.com/engine/install/ubuntu/) to install docker.
- Create container
  To pull the image: 
  ```sh
  docker pull starrychiao/hop-recurrent:v1
  ```
  To create the container:
  ```sh
  docker run -it --ipc host  --shm-size=1024m --gpus all --name your_name  --volume "your_directory":/root/mount/Matterport3DSimulator starrychiao/hop-recurrent:v1
  ```
- Set up
  ```sh
  docker start "your container id or name"
  docker exec -it "your container id or name" /bin/bash
  cd /root/mount/Matterport3DSimulator
  ```

### Data Preparation

Please follow the instructions below to prepare the data in directories:

- MP3D navigability graphs: `connectivity`
    - Download the [connectivity maps ](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
- MP3D image features: `img_features`
    - Download the [Scene features](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1) (ResNet-152-Places365).
- REVERIE data: `data_v2`
    - Download the [REVERIE data](https://github.com/YuankaiQi/REVERIE/tree/master/tasks/REVERIE/data_v2).
    - Download the object features [reverie_obj_feats_v2.pkl](https://drive.google.com/file/d/1zwV3QDPUVt7YmBNqTaCdS6v01U4b6p7M/view?usp=sharing)

- After downloading data you should see the following folder structure:
```
    ├── data_v2
    │   └── BBoxS
    │       └── reverie_obj_feats_v2.pkl
    │   └── BBoxes_v2
    ├── REVERIE_train.json
    ├── REVERIE_val_seen.json
    ├── REVERIE_val_unseen.json
    ├── REVERIE_test.json
    └── objpos.jsn
```

### Initial HOP weights
- Pre-trained HOP weights: `load/hop`
  - Download the `pytorch_model.bin` from [here](https://drive.google.com/drive/folders/1RtGij0T8__xrlhmVjFWqbQW2NYrcjK-R?usp=sharing).

### Training
```bash
bash run/agent.bash
```
### Evaluating
- To generate `submit_test.json`
```bash
bash run/test.bash
```
- You can also evaluate results on REVERIE seen and REVERIE unseen splits.
```bash
python ./r2r_src/eval.py
```

