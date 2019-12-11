#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:10:33 2019

@author: rj299
"""

import numpy as np
import os
import scipy.io as spio
import pandas as pd

#%%
data_behav_root = '/home/rj299/scratch60/mdm_analysis/data_behav'
out_root = '/home/rj299/scratch60/mdm_analysis/output'
# data_behav_root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Behavioral Analysis\PTB Behavior Log'

# subjects for imaging analysis
sub_num = [2073, 2550, 2582, 2583, 2584, 2585, 2588, 2592, 2593, 2594, 2596, 2597, 2598, 2599, 2600, 2624, 
           2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666]
#sub_nums = [2599, 2661]
    
#%% read parameters to calculate SV
par = pd.read_csv(os.path.join(data_behav_root, 'par_09300219.csv'))

#%% calculate SV
def ambig_utility(sub_id, par, p, a, obj_val, domain, model):
    '''
    Calcualte subjective value based on model
    For a list of trials
    
    Input:
        sub_id: subject id
        par: panda data frame of all subjects' parameter fits
        p: probability of lotteries, vector
        a: ambiguity of lotteries, vector
        obj_val: objective value of lottery pary-offs, vector
        domain_idx: domian indes, 1-medical, 0-monetary
        model: named of the subjective value model
        
    Output:
        sv: subjective values of lotteries, vector
    '''
    
    if domain == 'Med':
        domain_idx = 1
    elif domain == 'Mon':
        domain_idx = 0
        
    par_sub = par[(par.id == sub_id) & (par.is_med == domain_idx)]
    
    beta = par_sub.iloc[0]['beta']
    val1 = par_sub.iloc[0]['val1']
    val2 = par_sub.iloc[0]['val2']
    val3 = par_sub.iloc[0]['val3']
    val4 = par_sub.iloc[0]['val4']
    
    val = np.zeros(obj_val.shape)
    val[obj_val == 5] = val1
    val[obj_val == 8] = val2
    val[obj_val == 12] = val3
    val[obj_val == 25] = val4
    
    
    if model == 'ambigSVPar':       
        sv = (p - beta * a/2) * val
        
    ref_sv = np.ones(obj_val.shape) * val1
        
    return sv, ref_sv


#%%
def _todict(matobj):
    '''
    Author: Or
    
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def _check_keys(dict):
    '''
    Author: Or
    
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict 

def loadmat(filename):
    '''
    Author: Or
    
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

#%%
def readConditions(subNum, domain, matFile, x): # takes name of file and when to begin (i.e. first block is zero. second is 21 etc.)
    """ read condition onset and duration
    Author: Or
    
    Parameters
    -------------
    subNum: subject id
    domain: domain name, 'Med' or 'Mon'
    matFile: filename
    x: trial index at the begining of each block
    
    Return
    -------------
    events
    """
    
    metaData = loadmat(matFile)    
    # get the key names for the data, as half is 'Datamed', half is 'Datamon'
    data_keyname = list(metaData.keys())[3]    
    # trial number per block
    trial_num = 21 
    
    timeStamp = []
    condition = []
    events = []
    resultsArray = []
    duration = []
    resp_array =[]
    resp_onset = []
   
    ambigs = metaData[data_keyname]['ambigs']
    probs = metaData[data_keyname]['probs']
    vals = metaData[data_keyname]['vals']
    svs, ref_svs = ambig_utility(subNum, par, probs, ambigs, vals, domain, 'ambigSVPar')
    choice = metaData[data_keyname]['choice']
    refside = metaData[data_keyname]['refSide']
    
    # calculate response from choice and refside
    resp = np.ones(choice.shape) # 1-choose lottery
    resp[choice == refside] = 0 # 0-choose reference
    resp[choice == 0] = 2 # 2-no respone
    
    # x= 0 # where to start
    for i in range(x, x+trial_num):
        #a= metaData['Data']['trialTime'][i]
        #b = vars(a)
        
        # trial onset 
        resultsArray = vars(metaData[data_keyname]['trialTime'][i])['trialStartTime'] - vars(metaData[data_keyname]['trialTime'][x])['trialStartTime']
        timeStamp.append(int(round((3600*resultsArray[3] + 60*resultsArray[4] + resultsArray[5])))) # using int and round to round to the close integer. 
        
        # response onset
        resp_array = vars(metaData[data_keyname]['trialTime'][i])['feedbackStartTime'] - vars(metaData[data_keyname]['trialTime'][x])['trialStartTime']
        resp_onset.append(int(round((3600*resp_array[3] + 60*resp_array[4] + resp_array[5])))) # using int and round to round to the close integer.
        
        duration.append(6)
       
        if ambigs[i] == 0:
            condition.append('risk')
        else:
            condition.append('amb')
    
    events= pd.DataFrame({'trial_type':condition, 'onset':timeStamp, 'duration':duration, 
                          'probs': probs[range(x, x+trial_num)], 'ambigs': ambigs[range(x, x+trial_num)], 'vals': vals[range(x, x+trial_num)], 
                          'svs': np.round(svs[range(x, x+trial_num)], 3), 'ref_svs': np.round(ref_svs[range(x, x+trial_num)], 3), 
                          'resp': resp[range(x, x+trial_num)],
                          'resp_onset': resp_onset})[1:] # building data frame from what we took. Removing first row because its not used. 
    return events



def organizeBlocks(subNum):
    # Read both mat files (first timestamp)
    # check first block of each day. 
    # check thrird of each day
    # sort
    trial_num = 21
    
    orderArray = []
    
    mat_med_name = os.path.join(data_behav_root, 'subj%s' %subNum, 'MDM_MED_%s.mat' %subNum)    
    mat_mon_name = os.path.join(data_behav_root, 'subj%s' %subNum, 'MDM_MON_%s.mat' %subNum)    
    
#     matFileLoss = '/media/Drobo/Levy_Lab/Projects/R_A_PTSD_Imaging/Data/Behavior data/Behavior_fitpar/Behavior data fitpar_091318/RA_LOSS_%s_fitpar.mat'%subNum
#     matFileGain = '/media/Drobo/Levy_Lab/Projects/R_A_PTSD_Imaging/Data/Behavior data/Behavior_fitpar/Behavior data fitpar_091318/RA_GAINS_%s_fitpar.mat'%subNum
    metaDataMed = loadmat(mat_med_name)
    data_med_keyname = list(metaDataMed.keys())[3]
    metaDataMon = loadmat(mat_mon_name)
    data_mon_keyname = list(metaDataMon.keys())[3]
    
    # trial start time of the 1st and the 3rd block in each domain
    a= {'1stMed':list(vars(metaDataMed[data_med_keyname]['trialTime'][0])['trialStartTime']), '3rdMed':list(vars(metaDataMed[data_med_keyname]['trialTime'][trial_num*2])['trialStartTime']), '1stMon':list(vars(metaDataMon[data_mon_keyname]['trialTime'][0])['trialStartTime']), '3rdMon':list(vars(metaDataMon[data_mon_keyname]['trialTime'][trial_num*2])['trialStartTime'])}
    # sort by trial start time
    s = [(k, a[k]) for k in sorted(a, key=a.get, reverse=False)]
    for k, v in s:
        print (k, v)
        orderArray.append(k)
    
    totalEvent = []
    for n in orderArray:
        print (n)
        if n=='1stMed':
            # run Med mat file on readConcitions function on first two blocks (i.e. 0, 21)
            print (n)
            for x in [0,trial_num]:
                event = readConditions(subNum, 'Med', mat_med_name, x)
                event['condition'] = 'Med'
                totalEvent.append(event)
        elif n=='1stMon':
            # run Mon mat file on readCondition function
            print (n)
            for x in [0,trial_num]:
                event = readConditions(subNum, 'Mon', mat_mon_name, x)
                event['condition'] = 'Mon'
                totalEvent.append(event)
        elif n=='3rdMed':
            print (n)
            for x in [trial_num*2, trial_num*3]:
                event = readConditions(subNum, 'Med', mat_med_name, x)
                event['condition'] = 'Med'
                totalEvent.append(event)
        elif n=='3rdMon':
            # run Mon from 3rd block
            print (n)
            for x in [trial_num*2, trial_num*3]:
                event = readConditions(subNum, 'Mon', mat_mon_name, x)
                event['condition'] = 'Mon'
                totalEvent.append(event)
        else:
            print ('The condition ' + n + ' is not clear.')
        
        # the end result is an array of data sets per each run (i.e. block) - called totalEvent
    return totalEvent

#%% test 
sub_id = 2588
mat_med_name = os.path.join(data_behav_root, 'subj%s' %sub_id, 'MDM_MED_%s.mat' %sub_id)

mat_med_name
behav_med = loadmat(mat_med_name)
list(behav_med.keys())[3]
behav_med['Datamed'].keys()

sub_id = behav_med['Datamed']['observer']
probs = behav_med['Datamed']['probs']
ambigs = behav_med['Datamed']['ambigs']
vals = behav_med['Datamed']['vals']
trialTime = behav_med['Datamed']['trialTime']

choice = behav_med['Datamed']['choice']
print(choice)
print(choice.shape)

behav_med['Datamed']['trialTime'][0]._fieldnames
behav_med['Datamed']['trialTime'][0].__dict__['trialStartTime']
vars(behav_med['Datamed']['trialTime'][0])
list(range(0, 21))    


#%%
totalEvent_sub = organizeBlocks(2588)
totalEvent_sub[0]

#%%
# read conditions and write into csv files

for sub_id in sub_num:
    totalEvent_sub = organizeBlocks(sub_id)
    # write into csv
    
    for task_id in range(8):
        pd.DataFrame(totalEvent_sub[task_id]).to_csv(os.path.join(out_root, 'event_files', 'sub-' + str(sub_id)+ '_task-' +str(task_id+1) + '_cond_v3.csv'), 
                          index = False, sep = '\t')    