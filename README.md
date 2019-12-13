# General Nipype pipeline for task-based GLM fMRI analysis
This repository contains python scripts for running fMRI GLM analysis, both first and second levels, through Nipype. 

(Todo: 

SPM is used currently to fit GLM, but we also have the FSL scripts ready, and it should be added.

Piplines for including parameric modulater, and RSA are also ready. Need to be added.)

**Prerequisite**: fMRI data should already be preprocessed and in BIDS format. Behvaior data ready for creating design matrix.

## Steps:

**Step1:** Create event files for design matrix. Run create_event_files.py

**Step2:** fMRI scans in order. If necessary, run rename_imaging_files.py to rename scans.

**Step3:** First-level analysis, which is to fit GLM to each participant. Run the pipleline spm_glm_firstlevel.py

**Step4:** Second-level analysis, which is to analysis group-level contrast. Run the pipeline spm_glm_secondlevel.py

**Step5:** Visualize GLM results by running jupyter norebook visualize_glm_restuls.ipynb. Can also do further map thresholding and saving whole-brain analysis ROIS.

*If running on cluster in batch jobs, run_spm_glm_firstlevel.sh and run_spm_glm_secondlevel.sh are for submitting batch jobs.*

## Level1 pipeline flow:

###Images

Image:

![](https://https://github.com/LevyDecisionNeuroLab/fmri_task_glm/blob/master/graph.png)

> Workflow graph
