# General Nipype pipeline for task-based GLM fMRI analysis
This repository contains python scripts for running fMRI GLM analysis, both first and second levels, through Nipype. 

Scripts of using both FSL and SPM to fit GLM are included, but the two methods give slightly different outpus. Could choose either one to run.

## Prerequisites: 

+ fMRI data preprocessed and in BIDS format.
+ Behvaior data ready for creating design matrix.
+ Python environment ready, the following package installed: 
  + Basics: numpy, os, pandas, scipy, glob, matplotlib, seaborn
  + Neuroimaging: Nipype, Nilearn, Nibabel
  + Advanced stats and machine learning (not required for just running GLM): Brainiak, scikit-learn
+ FSL and SPM (MATLAB) installed.

## References:

Nipype tutorials: 

+ [Understand pipeline, node, iterfield, iterables, datasink](https://nipype.readthedocs.io/en/0.11.0/users/pipeline_tutorial.html) 

+ [How to specify design matrix](https://nipype.readthedocs.io/en/0.11.0/users/model_specification.html)

Poldrack lab resources: [Common workflows](https://github.com/poldracklab/niworkflows/tree/master/niworkflows)


## Steps:

+ **Step1:** Create event files for design matrix. Run ___create_event_files.py___.

+ **Step2:** fMRI scans in order. If necessary, run ***rename_imaging_files.py*** to rename scans.

+ **Step3:** First-level analysis, which is to fit GLM to each participant. Run the pipleline ***spm_glm_firstlevel.py***, ***fsl_glm_firstlevel.py***. Variations of GLM depend on different ways to set up the design matrix. The basic design matrix includes just binary preditors of trial types, and the response (if any, usually a brief button press) could be modeled as impulse. On top of this, parametric modulators could be added to the binary predictors (***spm_pmod_glm_firstlevel.py***). GLM could also be conducted before doing RSA (representational similary analysis), in which case each trial condtion for computing RDMs (representation dissimilarity matrix) is modeled as a predictor in the GLM, and no spatial smoothing is applied (***spm_rsa_glm_firstlevel.py***).

+ **Step4:** Second-level analysis, which is to analysis group-level contrast. Run the pipeline ***spm_glm_secondlevel.py***, ***fsl_glm_secondlevel.py***.

+ **Step5:** Visualize GLM results by running jupyter norebook ***visualize_glm_restuls.ipynb***. Can also do further map thresholding and saving whole-brain analysis ROIS.

+ **Running on cluster:** If running on cluster in batch jobs, ***run_spm_glm_firstlevel.sh*** and ***run_spm_glm_secondlevel.sh*** are for submitting batch jobs.

## Level1 pipeline flow chart:

![Level1 flowchart](https://github.com/LevyDecisionNeuroLab/fmri_task_glm/blob/master/graphs/graph.png)

> For basic GLM


![Level1 flowchart for RSA GLM](https://github.com/LevyDecisionNeuroLab/fmri_task_glm/blob/master/graphs/graph_rsa.png)
> For RSA GLM (no spatial smoothing!)

## Level2 pipeline flow chart:

![Level2 flowchart](https://github.com/LevyDecisionNeuroLab/fmri_task_glm/blob/master/graphs/graph_l2.png)
