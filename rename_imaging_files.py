#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:53:24 2019

@author: rj299
"""

# rename task id
import os
import glob

data_root = '/home/rj299/project/mdm_analysis/data_rename'
#%%
# functions for changing file names

# rename all files by adding run number determined by session and task number during scanning
def addRunNum(directory): 
    """ Add scan run numbers to file name
    
    Parameters
    --------------
    directory: directory for a subject, contains data for all runs
    
    """
    
    os.chdir(directory)
    
    # get sorted task number from the directory
    task_num_all = getTaskNum(directory)
    
    # add run number and rename
    for filename in os.listdir(directory):
        # get task number
        task_num = int(filename.split('_task-task')[1].split('_')[0])
        
        # get the run number based on all the task number in the directory
        run_count = task_num_all.index(task_num) + 1

        
        filename_new = filename.split('_task-task%s' %task_num)[0] + '_task-%s' %run_count + filename.split('_task-task%s' %task_num)[1]

        os.rename(filename, filename_new)  
        print(filename_new)

# get all task numbers for ses one
def getTaskNum(directory):
    """ Get all the task number for a session
     
    Parameters
    -----------------
    directory: data directory for a subject
    
    Return
    -----------------
    task_num: sorted task number for each session
    """
    file_ses = glob.glob('sub-*_ses-1_task*_bold.nii.gz')
    
    task_num = []
    
    for file in file_ses:
        task_id = file.split('_task-task')[1].split('_space')[0]
        task_num.append(int(task_id))
    
    task_num.sort()
    
    return task_num

#%% NEEDS to be run only once
    
# rename files and add run number in the file name
# needs running only ONCE
sub_fold = ['/home/rj299/project/mdm_analysis/data_rename/sub-2654',
            '/home/rj299/project/mdm_analysis/data_rename/sub-2658']

for fold in sub_fold:
    if fold != '/home/rj299/project/mdm_analysis/data_rename/sub-2582':
        fold_func = os.path.join(fold, 'ses-1', 'func')
        addRunNum(fold_func)    

#%% subject 2582 only
# rename files and add run number in the file name
# subject 2582 only, because the files are named in a different way
# rename all files by adding run number determined by session and task number during scanning

# run ONLY ONCE

def addRunNum_2582(directory): 
     """ Add scan run numbers to file name
    
     Parameters
     --------------
     directory: directory for a subject, contains data for all runs
    
     """
     os.chdir(directory)
    
     # get sorted task number from the directory
     task_num_all = getTaskNum_2582(directory)
    
     # add run number and rename
     for filename in os.listdir(directory):
         # get task number
         task_num = int(filename.split('_task-')[1].split('_')[0])
        
         # get the run number based on all the task number in the directory
         run_count = task_num_all.index(task_num) + 1

        
         filename_new = filename.split('_task-%s' %task_num)[0] + '_task-task-%s' %run_count + filename.split('_task-%s' %task_num)[1]

         os.rename(filename, filename_new)  
         print(filename_new)

 # get all task numbers for ses one
def getTaskNum_2582(directory):
     """ Get all the task number for a session
    
     Parameters
     -----------------
     directory: data directory for a subject
    
     Return
     -----------------
     task_num: sorted task number for each session
     """
     
     os.chdir(directory)
     
     file_ses = glob.glob('sub-*_ses-1_task*_bold.nii.gz')
    
     task_num = []
    
     for file in file_ses:
         task_id = file.split('_task-')[1].split('_space')[0]
         task_num.append(int(task_id))
    
     task_num.sort()
    
     return task_num

#%%
# subject 2582 has a weird way of naming files, need to run these two steps sequentially
# step 1, add run number, by adding an extra stirng of 'task' to prevent renaming wrong files
addRunNum_2582(os.path.join(data_root, 'sub-2582','ses-1','func')) 
# step 2, get rid of 'task' 
addRunNum(os.path.join(data_root, 'sub-2582','ses-1','func'))
#%%
task_num_temp = getTaskNum_2582(os.path.join(data_root, 'sub-2582','ses-1','func')) 
