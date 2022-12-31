#!/bin/sh
#SBATCH --job-name=train_gan_256
#SBATCH -o ./logs/%x-%j.log

# for future: add batch jobs

# print hostname
hostname

# add the pip installer to it
pip install -r /home/ds21m011/mi/requirements.txt
#pip install --upgrade numpy

# run the script
srun python ./mi/GAN_Model/GAN_cluster.py 3000
