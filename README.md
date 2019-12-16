# General Nipype pipeline for task-based GLM fMRI analysis
This repository contains python scripts for running fMRI GLM analysis, both first and second levels, through Nipype. 

(Todo: 

SPM is used currently to fit GLM, but we also have the FSL scripts ready, and it should be added.

Piplines for including parameric modulater, and RSA are also ready. Need to be added.)

## Prerequisites: 

+ fMRI data preprocessed and in BIDS format.
+ Behvaior data ready for creating design matrix.
+ Python environment ready, the following package installed: 
  + Basics: numpy, os, pandas, scipy, glob, matplotlib, seaborn
  + Neuroimaging: Nipype, Nilearn, Nibabel
  + Advanced stats and machine learning (not required for just running GLM): Brainiak, scikit-learn
+ FSL and SPM (MATLAB) installed.

## References:

Nipype tutorials: [Understand pipeline, node, iterfield, iterables, datasink](https://nipype.readthedocs.io/en/0.11.0/users/pipeline_tutorial.html) 

Poldracklab: [Common workflows](https://github.com/poldracklab/niworkflows/tree/master/niworkflows)


## Steps:

+ **Step1:** Create event files for design matrix. Run create_event_files.py

+ **Step2:** fMRI scans in order. If necessary, run rename_imaging_files.py to rename scans.

+ **Step3:** First-level analysis, which is to fit GLM to each participant. Run the pipleline spm_glm_firstlevel.py

+ **Step4:** Second-level analysis, which is to analysis group-level contrast. Run the pipeline spm_glm_secondlevel.py

+ **Step5:** Visualize GLM results by running jupyter norebook visualize_glm_restuls.ipynb. Can also do further map thresholding and saving whole-brain analysis ROIS.

*If running on cluster in batch jobs, run_spm_glm_firstlevel.sh and run_spm_glm_secondlevel.sh are for submitting batch jobs.*

## Level1 pipeline flow chart:

![Level1 flowchart](https://github.com/LevyDecisionNeuroLab/fmri_task_glm/blob/master/graphs/graph.png)

## Level2 pipeline flow chart:

![Level2 flowchart](https://github.com/LevyDecisionNeuroLab/fmri_task_glm/blob/master/graphs/graph_l2.png)
