#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:49:09 2019

Reference: https://github.com/poldracklab/ds003-post-fMRIPrep-analysis/blob/master/workflows.py

@author: rj299
"""
import nipype.interfaces.io as nio  # Data i/o
from nipype.interfaces import spm
from nipype import Node, Workflow, MapNode
import nipype.interfaces.utility as util # utility
from nipype import SelectFiles
import os

from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/home/rj299/project/MATLAB/toolbox/spm12/') # set default SPM12 path in my computer. 

#%% Gourp analysis - based on SPM - should consider the fsl Randomize option (other script)
# OneSampleTTestDesign - creates one sample T-Test Design
onesamplettestdes = Node(spm.OneSampleTTestDesign(),
                         name="onesampttestdes")

# EstimateModel - estimates the model
level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                      name="level2estimate")

# EstimateContrast - estimates group contrast
level2conestimate = Node(spm.EstimateContrast(group_contrast=True),
                         name="level2conestimate")
cont1 = ['Group', 'T', ['mean'], [1]]
level2conestimate.inputs.contrasts = [cont1]

# Which contrasts to use for the 2nd-level analysis
contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004', 'con_0005',
                 'con_0006', 'con_0007', 'con_0008', 'con_0009']

subject_list = [2073, 2550, 2582, 2583, 2584, 2585, 2588, 2592, 2593, 2594, 
                2596, 2597, 2598, 2599, 2600, 2624, 2650, 2651, 2652, 2653, 
                2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 
                2664, 2665, 2666]

#subject_list = [2073, 2550, 2582, 2583, 2584, 2585, 2588, 2592, 2593, 2594, 
#                2596, 2597, 2598, 2600, 2624, 2650, 2651, 2652, 2653, 
#                2654, 2655, 2656, 2657, 2658, 2659, 2660, 2662, 2663, 
#                2664, 2665, 2666]

# Threshold - thresholds contrasts
level2thresh = Node(spm.Threshold(contrast_index=1,
                              use_topo_fdr=False,
                              use_fwe_correction=True, # here we can use fwe or fdr
                              extent_threshold=10,
                              height_threshold= 0.005,
                              extent_fdr_p_threshold = 0.05,
                              height_threshold_type='p-value'),
                              
                                   name="level2thresh")

 #Infosource - a function free node to iterate over the list of subject names
infosource = Node(util.IdentityInterface(fields=['contrast_id', 'subject_id']),
                  name="infosource")

infosource.iterables = [('contrast_id', contrast_list)]
infosource.inputs.subject_id = subject_list

# SelectFiles - to grab the data (alternative to DataGrabber)
templates = {'cons': os.path.join('/home/rj299/scratch60/mdm_analysis/output/imaging/Sink/1stLevel/_subject_id_{subject_id}/', 
                         '{contrast_id}.nii')}

selectfiles = MapNode(SelectFiles(templates,
                               base_directory='/home/rj299/scratch60/mdm_analysis/work/',
                               sort_filelist=True),
                   name="selectfiles", 
                   iterfield = ['subject_id'])

datasink = Node(nio.DataSink(base_directory='/home/rj299/scratch60/mdm_analysis/output/imaging/Sink/'),
                name="datasink")


l2analysis = Workflow(name='l2spm')

l2analysis.base_dir = '/home/rj299/scratch60/mdm_analysis/work/'

l2analysis.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                               ('subject_id', 'subject_id')]),

                    (selectfiles, onesamplettestdes, [('cons', 'in_files')]),
                    
                    (onesamplettestdes, level2estimate, [('spm_mat_file',
                                                          'spm_mat_file')]),
                    (level2estimate, level2conestimate, [('spm_mat_file',
                                                          'spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')]),
                    (level2conestimate, level2thresh, [('spm_mat_file',
                                                        'spm_mat_file'),
                                                       ('spmT_images',
                                                        'stat_image'),
                                                       ]),
                    (level2conestimate, datasink, [('spm_mat_file',
                        '2ndLevel_cluster.@spm_mat'),
                       ('spmT_images',
                        '2ndLevel_cluster.@T'),
                       ('con_images',
                        '2ndLevel_cluster.@con')]),
                    (level2thresh, datasink, [('thresholded_map',
                                               '2ndLevel_cluster.@threshold')]),
                                                        ])
#%%                                                     
l2analysis.run('MultiProc', plugin_args={'n_procs': 3})