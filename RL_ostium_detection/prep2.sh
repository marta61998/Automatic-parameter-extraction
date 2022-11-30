#!/bin/bash
#SBATCH -J Prep-RL
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=32g
#SBATCH -o /homedtic/msaiz/Automatic_ostium_detection/logs2/job-%N-%J.out
#SBATCH -e /homedtic/msaiz/Automatic_ostium_detection/logs2/job-%N-%J.err

source ~/anaconda3/bin/activate "";
conda activate ost_env_4;

python3 preprocessing_final.py --input /homedtic/msaiz/Automatic_ostium_detection/pre_segmentation_test --output /homedtic/msaiz/Automatic_ostium_detection/output_case_101 --landmark_GT 0 &&

cd '/homedtic/msaiz/Automatic_ostium_detection/DQN/DQN-code' &&

python3 DQN.py --load '/homedtic/msaiz/Automatic_ostium_detection/DQN/Models/LAA/03_LAA_Ostium/model-750000.data-00000-of-00001' --task 'eval' --files '/homedtic/msaiz/Automatic_ostium_detection/output_case_101/images.txt' '/homedtic/msaiz/Automatic_ostium_detection/output_case_101/landmark.txt' --csv_name '/homedtic/msaiz/Automatic_ostium_detection/output_case_101/results_job.csv' --not_randstar &&

cd '/homedtic/msaiz/Automatic_ostium_detection' &&

python3 Postprocessing.py --input_image /homedtic/msaiz/Automatic_ostium_detection/output_case_101/image_noflip --input_csv /homedtic/msaiz/Automatic_ostium_detection/output_case_101/results_job.csv --output /homedtic/msaiz/Automatic_ostium_detection/output_case_101/post --RAS2LPS 0 