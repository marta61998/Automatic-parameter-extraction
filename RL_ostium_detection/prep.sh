#!/bin/bash
#SBATCH -J Prep-RL
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32g
#SBATCH -o /homedtic/msaiz/Automatic_ostium_detection/logs2/job-%N-%J.out
#SBATCH -e /homedtic/msaiz/Automatic_ostium_detection/logs2/job-%N-%J.err

source ~/anaconda3/bin/activate "";
conda activate ost_env;


python3 Preprocessing.py --input /homedtic/msaiz/Automatic_ostium_detection/pre_segmentation --output /homedtic/msaiz/Automatic_ostium_detection/pre_segmentation_out  --landmark_GT 0