# switch color mode
tput setaf 2

# switch to the new software stack
source /cluster/apps/local/env2lmod.sh
echo "[EULER SETUP] switched to new software stack"

# load GCC 8.2 and Python 3.8.5
module load gcc/8.2.0  > /dev/null
module load python/3.8.5  > /dev/null

# load ffmpeg (required for video conversion)
module load ffmpeg

# load git module, fetch new index and show status
module load git/2.31.1
echo "[EULER SETUP] loaded required modules"
echo "[EULER SETUP] fetching new index from GitHub"
git fetch > /dev/null
echo "[EULER SETUP] git status reported the following:"
echo "======================================================================"
tput sgr0
git status
tput setaf 2
echo "======================================================================"

# exit and reset color mode back to default
echo "[EULER SETUP] setup complete"
tput sgr0
