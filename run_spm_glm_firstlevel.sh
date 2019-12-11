#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=log_%J.txt
#SBATCH --error=error_%J.err
#SBATCH --job-name=glm_resp_2073
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=22:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruonan.jia@yale.edu

module load miniconda
module load FSL
module load MATLAB/2018b
source activate py37_dev

python /home/rj299/scratch60/mdm_analysis/mdm_imaging_analysis/spm_glm_firstlevel.py
