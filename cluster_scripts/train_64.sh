#!/bin/sh
#SBATCH --job-name=train_gan_64
#SBATCH -o ./logs/%x-%j.out

# for future: add batch jobs

# print hostname
hostname

# add the pip installer to it
pip install -r /home/ds21m011/mi/requirements.txt
#pip install --upgrade numpy

# run the script
srun python ./mi/GAN_Model/gan_cluster_64.py 20000
