#!/bin/sh
#SBATCH --job-name=dense
#SBATCH -o ./logs/%x-%j.log
#SBATCH --gpus-per-node=1
#SBATCH -t 1-0
#SBATCH --mem=28000

# for future: add batch jobs

# print hostname
hostname

# add the pip installer to it
pip install -r /home/ds21m011/mi/densenet_requirements.txt
#pip install --upgrade numpy

# run the script
# args: epochs, file path, save path, checkpoint path, batch size
srun python ./mi/densenet/densenet_cluster.py 1000 64 400 64