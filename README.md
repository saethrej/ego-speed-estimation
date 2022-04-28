# Learning-Based Ego Vehicle Speed Estimation

This is the repository for the [3D Vision course](http://www.cvg.ethz.ch/teaching/3dvision/ "Link to the course website.") project of group 10, consisting of team members Jens Eirik Saethre, Christopher Raffl, and Livio Schläpfer.

## Table of Contents
- [Getting Started](#getting-started)
    - [Setting up the Repository](#setting-up-the-repository)
    - [Downloading the comma2k19 Dataset](#downloading-the-comma2k19-dataset)
    - [Pre-Processing the comma2k19 Dataset](#pre-processing-the-comma2k19-dataset)
- [Directory Structure](#directory-structure)
- [Reproducing Results](#reproducing-results)
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
We use the [comma2k19](https://github.com/commaai/comma2k19 "Link to the git repository") dataset from [comma.ai](https://comma.ai/ "link to comma.ai webpage") for this project. The full dataset is approximately 100GB in size and must be downloaded via Torrent from Academic Torrents [here](https://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb). Note that it is also possible to download only some of the chunks.

When downloaded, unzip the individual chunks in the `data/comma2k19' directory via the following command (replace X with chunk number):
```bash
unzip Chunk_X.zip -d /path/to/ego-speed-estimation/data/comma2k19/
```

### Pre-Processing the comma2k19 Dataset 
The videos in the comma2k19 dataset contains 2085 video sequences of 1 minute length and 1200 frames. The frame rate is thus 20 fps. The videos are stored as `.hevc` files, which cannot be read with standard libraries such as `torchvision`. Moreover, the vehicle's speed data has a higher temporal resolution than the camera's frame rate. 

We thus provide two auxiliary scripts to (a) convert the input videos to `.mp4` files, and (b) extract the correct speed values at each video frame. These can be run with the following commands:

```bash
# load necessary modules (only if running on Euler)
source scripts/setup_euler.sh

# convert videos to mp4 format (takes a few hours)
python scripts/convert_videos.py

# extract required speed values (takes a few seconds)
python scripts/extract_velocities.py
``` 

After that, there will be two files `video.mp4` and `speeds.npy` for every video sequence.

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
 ┃ ┃ ┃ ┃ ┃ ┗ 📜video.mp4
 ┣ 📂src
 ┃ ┣ 📂data_loader
 ┃ ┃ ┣ 📂datasets
 ┃ ┃ ┃ ┣ 📜commaAI.py
 ┃ ┃ ┃ ┗ 📜runner_datasets.py
 ┃ ┃ ┗ 📜default.py
 ┃ ┣ 📂initializer
 ┃ ┃ ┣ 📜initialization.py
 ┃ ┃ ┗ 📜initialization_dataset.py
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📜cnn_lstm.py
 ┃ ┃ ┗ 📜runner_models.py
 ┃ ┣ 📂preprocessing
 ┃ ┃ ┣ 📜default.py
 ┃ ┃ ┗ 📜runner_preprocessing.py
 ┃ ┣ 📜test_loop.py
 ┃ ┗ 📜train_loop.py
 ┣ 📜runner_testing.py
 ┗ 📜runner_training.py
```
</details>

Notes: tbd

## Reproducing Results

tbd.

## Deliverables

The proposal can be found [here](https://github.com/saethrej/ego-speed-estimation/raw/main/deliverables/proposal.pdf) and the final report can be found [here](https://github.com/saethrej/ego-speed-estimation/raw/main/deliverables/report.pdf).

## Authors

- [Jens Eirik Saethre](https://www.linkedin.com/in/saethrej/) (saethrej)
- [Christopher Raffl](https://www.linkedin.com/in/christopher-raffl/) (rafflc)
- [Livio Schläpfer](https://www.linkedin.com/in/livio-schl%C3%A4pfer-b34607179/) (livios)

All authors contributed equally to this project.
