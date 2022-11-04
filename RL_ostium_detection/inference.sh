#!/bin/bash
#SBATCH -J Predict-RL
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32g
#SBATCH -o /homedtic/msaiz/Automatic_ostium_detection/logs/job-%N-%J.out
#SBATCH -e /homedtic/msaiz/Automatic_ostium_detection/logs/job-%N-%J.err

source ~/anaconda3/bin/activate "";
conda activate ost_env_2;

cd '/homedtic/msaiz/Automatic_ostium_detection/DQN/DQN-code'

python3 DQN.py --load '/homedtic/msaiz/Automatic_ostium_detection/DQN/Models/LAA/03_LAA_Ostium/model-750000.data-00000-of-00001' --task 'eval' --files '/homedtic/msaiz/Automatic_ostium_detection/images_2.txt' '/homedtic/msaiz/Automatic_ostium_detection/landmark_2.txt' --csv_name '/homedtic/msaiz/Automatic_ostium_detection/results/results_job4.csv' --not_randstar