# Learning-Based Ego Vehicle Speed Estimation

This is the repository for the [3D Vision course](http://www.cvg.ethz.ch/teaching/3dvision/ "Link to the course website.") project of group 10, consisting of team members Jens Eirik Saethre, Christopher Raffl, and Livio Schläpfer.

## Table of Contents
- [Getting Started](#getting-started)
    - [Setting up the Repository](#setting-up-the-repository)
    - [Downloading the comma2k19 Dataset](#downloading-the-comma2k19-dataset)
    - [Pre-Processing the comma2k19 Dataset](#pre-processing-the-comma2k19-dataset)
    - [Additional Dependencies](#additional-dependencies)
- [Directory Structure](#directory-structure)
- [Training and Evaluation](#training-and-evaluation)
    - [Configuration Files](#configuration-files)
    - [Training](#training)
    - [Manual Evaluation](#manual-evaluation)
- [Deliverables](#deliverables)
- [Authors](#authors)

## Getting Started
### Setting up the Repository
Set-up the project repository by running the following:
```bash
# clone repository
git clone https://github.com/saethrej/ego-speed-estimation.git && cd ego-speed-estimation

# install requirements (possibly in a virtual environment)
pip install -r requirements.txt
```

### Downloading the comma2k19 Dataset
We use the [comma2k19](https://github.com/commaai/comma2k19 "Link to the git repository") dataset from [comma.ai](https://comma.ai/ "link to comma.ai webpage") for this project. The full dataset is approximately 100GB in size and must be downloaded via Torrent from Academic Torrents [here](https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb). Note that it is also possible to partially download the dataset by selecting dataset chunks.

When downloaded, unzip the individual chunks in the `data/comma2k19' directory via the following command (replace X with chunk number):
```bash
unzip Chunk_X.zip -d /path/to/ego-speed-estimation/data/comma2k19/
```

### Pre-Processing the comma2k19 Dataset 
The comma2k19 dataset consists of 2019 video sequences, each of 1 minute length, captured at 20 fps, and thus denote a sequence of 1200 frames. The raw videos are provided as `.hevc` files, which cannot be directly read with standard libraries such as `torchvision`. Moreover, the vehicle's speed data has a higher temporal resolution than the camera's frame rate. 

We thus provide three auxiliary scripts: (a) converts the input videos to `.mp4` files, (b) crops the initial frames to remove static elements such as the front of the car's cockpit and further resize them to 290 x 118 pixels, and (c) extracts the correct speed values at each video frame. To run the scripts use the following commands:

```bash
# load necessary modules (only if running on Euler)
source scripts/setup_euler.sh

# convert videos to mp4 format (takes a few hours)
python scripts/convert_videos.py

# crop and resize mp4 videos (takes a few hours)
python scripts/compress_videos.py

# extract required speed values (takes a few seconds)
python scripts/extract_velocities.py
``` 

The scripts will generate two files `video_comcro.mp4` and `speeds.npy` for every video sequence.

### Additional dependencies
For depth estimation we use a model from [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/) that is loaded with `torch.hub.load`. If this code is executed in an environment with restricted access to the Internet, it is not possible to load the model in this fashion. As a workaround, we manually copied the required model into a directory `/torch_hub`. Our code then automatically checks if this directory exists and, if this is the case, loads the content from there. To replicate this setup, perform the following steps:
- Execute the download script on a machine that has access to the Internet by running `python scripts/download_midas.py`
- The model will be downloaded and cached on the machine. This is usually in the directory `~/.cache/torch/hub`
- Create a new directory 'torch_hub' in the top level of this project on the machine with restricted access.
- Copy the content of your local cache into this directory by e.g. using FTP
- Rerun the script on the target machine to check if the model can be loaded correctly from the directory

## Directory Structure
<details><summary>Click here to show the structure of the directory.</summary>

```
📦project
 ┣ 📂config
 ┃ ┣ 📂dataset_config
 ┃ ┃ ┗ 📜commai-ai.yaml
 ┃ ┗ 📜default.yaml
 ┣ 📂data
 ┃ ┗ 📂comma2k19
 ┃ ┃ ┣ 📂Chunk_X
 ┃ ┃ ┃ ┣ 📂example-capture
 ┃ ┃ ┃ ┃ ┣ 📂example-sequence
 ┃ ┃ ┃ ┃ ┃ ┣ 📂raw-data-folders
 ┃ ┃ ┃ ┃ ┃ ┣ ...
 ┃ ┃ ┃ ┃ ┃ ┣ 📜speeds.npy
 ┃ ┃ ┃ ┃ ┃ ┗ 📜video_comcro.mp4
 ┣ 📂scripts
 ┃ ┣ 📜compress_video.py
 ┃ ┣ 📜convert_video.py
 ┃ ┣ 📜download_midas.py
 ┃ ┣ 📜extract_velocities.py
 ┃ ┗ 📜setup_euler.sh
 ┣ 📂src
 ┃ ┣ 📂data_loader
 ┃ ┃ ┣ 📂datasets
 ┃ ┃ ┃ ┣ 📜commaAI.py
 ┃ ┃ ┃ ┗ 📜runner_datasets.py
 ┃ ┃ ┗ 📜default.py
 ┃ ┣ 📂initializer
 ┃ ┃ ┣ 📜initialization_dataset.py
 ┃ ┃ ┗ 📜initialization.py
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📜bandari_baseline.py
 ┃ ┃ ┣ 📜dof_cnn_lstm.py
 ┃ ┃ ┣ 📜dof_cnn.py
 ┃ ┃ ┣ 📜dual_cnn_lstm.py
 ┃ ┃ ┗ 📜runner_models.py
 ┃ ┣ 📂preprocessing
 ┃ ┃ ┣ 📜default.py
 ┃ ┃ ┣ 📜optical_flow.py
 ┃ ┃ ┗ 📜runner_preprocessing.py
 ┃ ┣ 📜test_loop.py
 ┃ ┗ 📜train_loop.py
 ┣ 📜runner_testing.py
 ┗ 📜runner_training.py
```
</details>

### Directory contents

- /config: Configuration files
- /data: Data files
- /scripts: Helper scripts for converting and compressing the video frames, extracting the velocities from the data set, and setting up the required environment for them on euler
- /src/data_loader: Initializes data loader and loads data during training/testing
- /src/initializer: Initializes configuration and fixes seeds
- /src/models: Implementation of models and handles loading of them
- /src/preprocessing: Implementation of preprocessing and handles loading of them
- /src: Training and Testing loops
- /: Scripts for starting the pipeline for training and testing

## Training and Evaluation

### Configuration Files
There are two different configuration files: /config/dataset_config/commai-ai.yaml defines the configuration for the comma ai dataset and /config/default.yaml defines the configuration for the model itself. For a detailed description of the parameters available we refer to the comments in these files. Make sure to define everything appropriately before continuing to the next steps.

### Training

After defining the models and parameters in the configuration, training can be start by executing the following command within the initialized environment
```bash
python runner_training.py --test
```
The `--test` flag is optional and automatically triggers the evaluation after training has finished.
The `--local` flag defines if the paths specified under local or under euler should be used.

To issue a job that runs on the Euler cluster of ETH Zürich execute within the initialized environment
```bash
bsub -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W x:00 python runner_training.py --test
``` 
Note that the required amount of time depends on the selected configuration.

### Manual evaluation

Evaluation can be automatically triggered after training by appending the `--test` flag. To manually trigger the evaluation execute 
```bash
python runner_testing.py
```
The weights used for evaluation can either be specified by (a) appending `--weights [dir]` to the above command where [dir] is the name of the directory within the output_path directory and should be of the form 'run_2022-xx-xx_xx-xx-xx' or (b) defining it in the configuration file.


## Deliverables

The proposal can be found [here](https://github.com/saethrej/ego-speed-estimation/raw/main/deliverables/proposal.pdf) and the final report can be found [here](https://github.com/saethrej/ego-speed-estimation/raw/main/deliverables/report.pdf).

## Authors

- [Jens Eirik Saethre](https://www.linkedin.com/in/saethrej/) (saethrej)
- [Christopher Raffl](https://www.linkedin.com/in/christopher-raffl/) (rafflc)
- [Livio Schläpfer](https://www.linkedin.com/in/livio-schl%C3%A4pfer-b34607179/) (livios)

All authors contributed equally to this project.
