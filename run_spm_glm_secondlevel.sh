#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=spm_2st_log_%J.txt
#SBATCH --error=spm_2st_error_%J.err
#SBATCH --job-name=spm2st_33sub
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruonan.jia@yale.edu

module load miniconda
module load FSL
module load MATLAB/2018b
source activate py37_dev

python /home/rj299/scratch60/mdm_analysis/mdm_imaging_analysis/spm_glm_secondlevel.py
