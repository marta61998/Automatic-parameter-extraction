#!/bin/bash
#SBATCH -J Prep-RL
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32g
#SBATCH -o /homedtic/msaiz/Automatic_ostium_detection/logs2/job-%N-%J.out
#SBATCH -e /homedtic/msaiz/Automatic_ostium_detection/logs2/job-%N-%J.err

source ~/anaconda3/bin/activate "";
conda activate vmtk;

python3 Script_final.py --input_mesh /homedtic/msaiz/Automatic_ostium_detection/algoritmo_marta/case_101.stl --input_excel /homedtic/msaiz/Automatic_ostium_detection/output_case_101/post/ostium_pred.xlsx --output /homedtic/msaiz/Automatic_ostium_detection/output_case_101