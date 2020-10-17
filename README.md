# PrivacyGANs

- ArXiv: [Towards Generalized and Distributed Privacy-Preserving Representation Learning](https://arxiv.org/abs/2010.01792)
- Github: [PrivacyGANs](https://github.com/shams-sam/PrivacyGANs)
- Data: [Drive Link](https://drive.google.com/file/d/1h1brXcywHgxCEFKzjc0yu6WQMr8IRUvz/view?usp=sharing)

## Instructions

- **Datasets have to be downloaded individually as per regulations and copyrights.**
- Drive contains the model checkpoints, training histories, and corresponding plots.
- Data is in the same directory structure as required by project (paste in corresponding folders).
- docker integration is used to reduce the overhead of setting up environment.
- users are welcome to use non-docker environments on their own.
- prepopulated hyperparameters and training logs as well as pretrained models are made available for evaluation.


## Docker Setup

### To build the docker image
- replace `gpu` with `cpu` in `docker-dl-setup/docker-compose.yml` in case the system has no gpu
- script to build the docker image

```shell
cd docker-dl-setup
docker-compose build
```

### To run the docker container

```shell
./run-docker.sh
```

### To enter the docker container

```shell
docker exec -it eigan_devel bash
```

## Training
- all scripts are run from `*.sh` files in `scripts` folder
- change the hyperparameters, as in example scripts
- run the scripts inside the docker container

```shell
sh scipts/<mimic/mnist/titanic>/<script-name>.sh
```

## Source folder executions
```shell
cd src
sh sh/<script-name>.sh <expt-name>
```

## Comparison
- comparison scripts need editing of python scripts
- replace the names of the pre-populated training histories with the newly generated training histories after training to generate new plots and analysis.
