# HOP-REVERIE-Challenge

This respository is the code of [HOP](https://arxiv.org/abs/2203.11591) for REVERIE-Challenge. The code is based on [Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT). Thanks to [Yicong Hong](https://github.com/YicongHong) for releasing the Recurrent-VLN-BERT code.

## Prerequisites
### Installation
- Install docker.
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
* Please check [here](https://github.com/YuankaiQi/REVERIE) to download data.

### Initial HOP weights
- Pre-trained HOP weights: `load`
  - Download the `pytorch_model.bin` from [here](https://drive.google.com/drive/folders/1RtGij0T8__xrlhmVjFWqbQW2NYrcjK-R?usp=sharing).

### Training
```bash
bash run/train.bash
```
### Evaluating
The generated `submit_test.json`
```bash
bash run/test.bash
```

