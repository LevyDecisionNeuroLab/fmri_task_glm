# -*- coding: utf-8 -*-
"""
This script runs GLM for conductiong RSA.
The most important difference is that predictor definition is based on the RSA interest.
Each condition for calculating the RDM is modeled as a single predictor
Another important difference is that no spatial smoothing is applied, because RSA wants to look at activation patterns.
If smoothing is added, the voxel pattern correlation between conditions is very high

Computing RDM from ROIs is added at the end of the pipeline, 
but could be treated as a separate step.

"""
import os
import pandas as pd
import numpy as np
#%%
base_root = '/home/rj299/scratch60/mdm_analysis/'
data_root = '/home/rj299/scratch60/mdm_analysis/data_rename'
out_root = '/home/rj299/scratch60/mdm_analysis/output'

# base_root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Imaging Analysis\RA_PTSD_SPM'
# data_root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Imaging Analysis\data_rename'
# out_root = 'D:\Ruonan\Projects in the lab\MDM Project\Medical Decision Making Imaging\MDM_imaging\Imaging Analysis\output'

#%%
from nipype.interfaces import spm

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
#import nipype.algorithms.rapidart as ra  # artifact detection
import nipype.algorithms.modelgen as model  # model specification
#from nipype.algorithms.rapidart import ArtifactDetect
# from nipype.algorithms.misc import Gunzip
from nipype import Node, Workflow, MapNode
from nipype.interfaces import fsl

from nipype.interfaces.matlab import MatlabCommand

import nibabel as nib
from nilearn.input_data import NiftiMasker

#%%
MatlabCommand.set_default_paths('/home/rj299/project/MATLAB/toolbox/spm12/') # set default SPM12 path in my computer. 
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

data_dir = data_root
output_dir = os.path.join(out_root, 'imaging')
work_dir = os.path.join(base_root, 'work') # intermediate products

#subject_list = [2073, 2550, 2582, 2583, 2584, 2585, 2588, 2592, 2593, 2594, 
#           2596, 2597, 2598, 2599, 2600, 2624, 2650, 2651, 2652, 2653, 
#           2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 
#           2664, 2665, 2666]

#subject_list = [2073, 2550, 2582, 2583, 2584, 2585, 2588]
#subject_list = [2592, 2593, 2594, 2596, 2597, 2598, 2599, 2600, 2624, 2650, 2651, 2652, 2653, 2654, 2655, 2656]
subject_list = [2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666]

# task_id = [1,2]

tr = 1
# first sevetal scans to delete
del_scan = 10

# Map field names to individual subject runs.
# infosource = pe.Node(util.IdentityInterface(fields=['subject_id', 'task_id'],),
#                   name="infosource")

# infosource.iterables = [('subject_id', subject_list), 
#                         ('task_id', task_list)]

infosource = pe.Node(util.IdentityInterface(fields=['subject_id'],),
                  name="infosource")

infosource.iterables = [('subject_id', subject_list)]


#%%

#in_file = '/home/rj299/project/mdm_analysis/data_rename/sub-2073/ses-1/func/sub-2073_ses-1_task-3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
#events_file = '/home/rj299/project/mdm_analysis/output/event_files/sub-2073_task-3_cond.csv'
#regressors_file = '/home/rj299/project/mdm_analysis/data_rename/sub-2073/ses-1/func/sub-2073_ses-1_task-3_desc-confounds_regressors.tsv'

def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0, del_scan=10):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch
    
    # Process the events file
    events = pd.read_csv(events_file, sep=r'\s+')

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()
    
    regress_data = pd.read_csv(regressors_file, sep=r'\s+')
    np.savetxt(out_motion, regress_data[motion_columns].fillna(0.0).values[del_scan:,], '%g')
#     np.savetxt(out_motion, regress_data[motion_columns].fillna(0.0).values, '%g')
    
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    domain = list(set(events.condition.values))[0] # domain of this task run, should be only one, 'Mon' or 'Med'
    trial_types = list(set(events.trial_type.values))
#    outcome_levels = list(set(events.vals.values))
    outcome_levels = {'0': 5, '1': 8, '2': 12, '3': 25}
    conds = [domain + '_' + trial_type for trial_type in trial_types] # e.g. ['Med_risk', 'Med_ambig']
    conditions = []
    for cond in conds:
        for outcome_level in list(outcome_levels.keys()):
            conditions.append(cond + '_' + outcome_level)
    
    
    runinfo = Bunch(
        scans=in_file,
        conditions=conditions, # should be 8 conditions (2 x 4)
        # conditions = ['Med_amb', 'Med_risk', 'Mon_amb', 'Mon_risk'],
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:        
        
        event = events[(events.trial_type.str.match(condition[4:len(condition)-2])) & (events.vals == outcome_levels[condition[-1:]])]
        runinfo.onsets.append(np.round(event.onset.values - del_scan + 1, 3).tolist()) # take out the first several deleted scans
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))
            
        # if domain == condition[:3]:
        #     event = events[events.trial_type.str.match(condition[4:])]
        #     runinfo.onsets.append(np.round(event.onset.values - del_scan + 1, 3).tolist()) # take out the first several deleted scans
        #     runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        #     if 'amplitudes' in events.columns:
        #         runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        #     else:
        #         runinfo.amplitudes.append([amplitude] * len(event))
                
        # else: # empty conditions
        #     runinfo.onsets.append([])
        #     runinfo.durations.append([])
        #     runinfo.amplitudes.append([])
        
    # delete empty condition, if any   
    cond_idx = 0
    while cond_idx < len(runinfo.conditions):
        if not runinfo.onsets[cond_idx]:
            runinfo.conditions.pop(cond_idx)
            runinfo.onsets.pop(cond_idx)
            runinfo.durations.pop(cond_idx)
            runinfo.amplitudes.pop(cond_idx)
        else:
            cond_idx += 1

    # response predictor regardless of condition
    runinfo.conditions.append('Resp')
    
    # response predictor when there is a button press
    resp_mask = events.resp != 2    
    resp_onset= np.round(events.resp_onset.values[resp_mask] - del_scan + 1, 3).tolist()
    runinfo.onsets.append(resp_onset)
    runinfo.durations.append([0] * len(resp_onset))
    runinfo.amplitudes.append([amplitude] * len(resp_onset))             
            
    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values[del_scan:,].T.tolist()

    return runinfo, str(out_motion)

#r_temp, o_temp = _bids2nipypeinfo(in_file, events_file, regressors_file,
#                     regressors_names=None,
#                     motion_columns=None,
#                     decimals=3, amplitude=1.0, del_scan=10)
#%%
templates = {'func': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
             'mask': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'),
             'regressors': os.path.join(data_root, 'sub-{subject_id}', 'ses-1', 'func', 'sub-{subject_id}_ses-1_task-{task_id}_desc-confounds_regressors.tsv'),
             'events': os.path.join(out_root, 'event_files', 'sub-{subject_id}_task-{task_id}_cond_v3.csv')}

# Flexibly collect data from disk to feed into workflows.
selectfiles = pe.Node(nio.SelectFiles(templates,
                      base_directory=data_root),
                      name="selectfiles")
        
selectfiles.inputs.task_id = [1,2,3,4,5,6,7,8]        
        
# Extract motion parameters from regressors file
runinfo = MapNode(util.Function(
    input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names', 'motion_columns'],
    function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
    name='runinfo',
    iterfield = ['in_file', 'events_file', 'regressors_file'])

# Set the column names to be used from the confounds file
# reference a paper from podlrack lab
runinfo.inputs.regressors_names = ['std_dvars', 'framewise_displacement'] + \
                                  ['a_comp_cor_%02d' % i for i in range(6)]
                                  

runinfo.inputs.motion_columns   = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2'] + \
                                  ['trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2'] + \
                                  ['trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2'] + \
                                  ['rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2'] + \
                                  ['rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2'] + \
                                  ['rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']


#%%
# gunzip = MapNode(Gunzip(), name='gunzip', iterfield=['in_file'])


# delete first several scans
# def extract_all(in_files):
#     from nipype.interfaces import fsl
#     roi_files = []
#     for in_file in in_files:
#         roi_file = fsl.ExtractROI(in_file = in_file, t_min = 10, t_size = -1, output_type = 'NIFTI')
#         roi_files.append(roi_file)  
#     return roi_files

        
# extract = pe.Node(util.Function(
#         input_names = ['in_files'],
#         function = extract_all, output_names = ['roi_files']),
#         name = 'extract')
        
extract = pe.MapNode(fsl.ExtractROI(), name="extract", iterfield = ['in_file'])
extract.inputs.t_min = del_scan
extract.inputs.t_size = -1
extract.inputs.output_type='NIFTI'

# set contrasts, depend on the condition
contrasts = []
all_conditions = ['Med_amb_0', 'Med_amb_1', 'Med_amb_2', 'Med_amb_3',
          'Med_risk_0', 'Med_risk_1', 'Med_risk_2', 'Med_risk_3',
          'Mon_amb_0', 'Mon_amb_1', 'Mon_amb_2', 'Mon_amb_3',
          'Mon_risk_0', 'Mon_risk_1', 'Mon_risk_2', 'Mon_risk_3']


for (cond_idx, single_condition) in enumerate(all_conditions):
    contrast_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    contrast_vector[cond_idx] = 1
    contrasts.append([single_condition, 'T',
                      all_conditions,
                      contrast_vector])
    
# no need to include the response contrast because it is not interest of analysis
# if included, the compute_roi_rdm needs to be adjusted, becasue it now takes all 
# contrasts passed from contrast estimate and iterate through. Reponse contrast should bot be included
#contrasts.append(['Response','T', ['Resp'], [1]])
    
# e.g. a single contrast should be:
#cont1 = ['Med_amb_0', 'T', 
#         ['Med_amb_0', 'Med_amb_1', 'Med_amb_2', 'Med_amb_3',
#          'Med_risk_0', 'Med_risk_1', 'Med_risk_2', 'Med_risk_3',
#          'Mon_amb_0', 'Mon_amb_1', 'Mon_amb_2', 'Mon_amb_3',
#          'Mon_risk_0', 'Mon_risk_1', 'Mon_risk_2', 'Mon_risk_3'], 
#         [1, 0, 0, 0,
#          0, 0, 0, 0,
#          0, 0, 0, 0,
#          0, 0, 0, 0]]
         

#%%

modelspec = Node(interface=model.SpecifySPMModel(), name="modelspec") 
modelspec.inputs.concatenate_runs = False
modelspec.inputs.input_units = 'scans' # supposedly it means tr
modelspec.inputs.output_units = 'scans'
#modelspec.inputs.outlier_files = '/media/Data/R_A_PTSD/preproccess_data/sub-1063_ses-01_task-3_bold_outliers.txt'
modelspec.inputs.time_repetition = 1.  # make sure its with a dot 
modelspec.inputs.high_pass_filter_cutoff = 128.

level1design = pe.Node(interface=spm.Level1Design(), name="level1design") #, base_dir = '/media/Data/work')
level1design.inputs.timing_units = modelspec.inputs.output_units
level1design.inputs.interscan_interval = 1.
level1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
level1design.inputs.model_serial_correlations = 'AR(1)'

# create workflow
wfSPM_rsa = Workflow(name="l1spm_resp_rsa_nosmooth", base_dir=work_dir)
wfSPM_rsa.connect([
        (infosource, selectfiles, [('subject_id', 'subject_id')]),
        (selectfiles, runinfo, [('events','events_file'),('regressors','regressors_file')]),
        (selectfiles, extract, [('func','in_file')]),
        (extract, runinfo, [('roi_file','in_file')]),
        (extract, modelspec, [('roi_file', 'functional_runs')]),   
        (runinfo, modelspec, [('info', 'subject_info'), ('realign_file', 'realignment_parameters')]),
        
        ])
wfSPM_rsa.connect([(modelspec, level1design, [("session_info", "session_info")])])

#%%
level1estimate = pe.Node(interface=spm.EstimateModel(), name="level1estimate")
level1estimate.inputs.estimation_method = {'Classical': 1}

contrastestimate = pe.Node(
    interface=spm.EstimateContrast(), name="contrastestimate")
#contrastestimate.inputs.contrasts = contrasts
contrastestimate.overwrite = True
contrastestimate.config = {'execution': {'remove_unnecessary_outputs': False}}
contrastestimate.inputs.contrasts = contrasts                                                   
                                                   

wfSPM_rsa.connect([
         (level1design, level1estimate, [('spm_mat_file','spm_mat_file')]),
         (level1estimate, contrastestimate,
            [('spm_mat_file', 'spm_mat_file'), ('beta_images', 'beta_images'),
            ('residual_image', 'residual_image')]),
    ])

#%% Adding data sink
########################################################################
# Datasink
datasink = Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_resp_rsa_nosmooth')),
                                         name="datasink")
                       
wfSPM_rsa.connect([
        (level1estimate, datasink, [('beta_images',  '1stLevel.@betas.@beta_images'),
                                    ('residual_image', '1stLevel.@betas.@residual_image'),
                                    ('residual_images', '1stLevel.@betas.@residual_images'),
                                    ('SDerror', '1stLevel.@betas.@SDerror'),
                                    ('SDbetas', '1stLevel.@betas.@SDbetas'),
                ])
        ])
    
wfSPM_rsa.connect([
       # here we take only the contrast ad spm.mat files of each subject and put it in different folder. It is more convenient like that. 
       (contrastestimate, datasink, [('spm_mat_file', '1stLevel.@spm_mat'),
                                              ('spmT_images', '1stLevel.@T'),
                                              ('con_images', '1stLevel.@con'),
                                              ('spmF_images', '1stLevel.@F'),
                                              ('ess_images', '1stLevel.@ess'),
                                              ])
        ])

#%% Compute ROI RDM
    
def compute_roi_rdm(in_file,
                    stims,
                    all_masks):
    
    from pathlib import Path
    from nilearn.input_data import NiftiMasker
    import numpy as np
    import nibabel as nib
    
    rdm_out = Path('roi_rdm.npy').resolve()
    stim_num = len(stims)
    
    # dictionary to store rdms for all rois
    rdm_dict = {}
    
    # loop over all rois
    for mask_name in all_masks.keys():
        mask = all_masks[mask_name]
        masker = NiftiMasker(mask_img=mask)
        
        # initiate matrix
        spmt_allstims_roi= np.zeros((stim_num, np.sum(mask.get_data())))
            
        for (stim_idx, spmt_file) in enumerate(in_file):
            spmt = nib.load(spmt_file)
          
            # get each condition's beta
            spmt_roi = masker.fit_transform(spmt)
            spmt_allstims_roi[stim_idx, :] = spmt_roi
        
        # create rdm
        rdm_roi = 1 - np.corrcoef(spmt_allstims_roi)
        
        rdm_dict[mask_name] = rdm_roi
        
    # save    
    np.save(rdm_out, rdm_dict)
    
    return str(rdm_out)



get_roi_rdm = Node(util.Function(
    input_names=['in_file', 'stims', 'all_masks'],
    function=compute_roi_rdm, 
    output_names=['rdm_out']),
    name='get_roi_rdm',
    )    
    
get_roi_rdm.inputs.stims = {'01': 'Med_amb_0', '02': 'Med_amb_1', '03': 'Med_amb_2', '04': 'Med_amb_3',
                            '05': 'Med_risk_0', '06': 'Med_risk_1', '07': 'Med_risk_2', '08': 'Med_risk_3', 
                            '09': 'Mon_amb_0', '10': 'Mon_amb_1', '11': 'Mon_amb_2', '12': 'Mon_amb_3',
                            '13': 'Mon_risk_0', '14': 'Mon_risk_1', '15': 'Mon_risk_2', '16': 'Mon_risk_3'}

# Masker files
maskfile_vmpfc = os.path.join(output_dir, 'binConjunc_PvNxDECxRECxMONxPRI_vmpfc.nii.gz')
maskfile_vstr = os.path.join(output_dir, 'binConjunc_PvNxDECxRECxMONxPRI_striatum.nii.gz')
maskfile_roi1 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p005_roi1.nii.gz')
maskfile_roi2 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p005_roi2.nii.gz')
maskfile_roi3 = os.path.join(output_dir, 'none_glm_Med_Mon_TFCE_p005_roi3.nii.gz')

maskfiles = {'vmpfc': maskfile_vmpfc, 
             'vstr': maskfile_vstr, 
             'med_mon_1': maskfile_roi1, 
             'med_mon_2': maskfile_roi2, 
             'med_mon_3': maskfile_roi3}

# roi inputs are loaded images
get_roi_rdm.inputs.all_masks = {key_name: nib.load(maskfiles[key_name]) for key_name in maskfiles.keys()}


wfSPM_rsa.connect([
        (contrastestimate, get_roi_rdm, [('spmT_images', 'in_file')]),
        ])

#%% data sink rdm
# Datasink
datasink_rdm = Node(nio.DataSink(base_directory=os.path.join(output_dir, 'Sink_resp_rsa_nosmooth')),
                                         name="datasink_rdm")
                       

wfSPM_rsa.connect([
        (get_roi_rdm, datasink_rdm, [('rdm_out', 'rdm.@rdm')]),
        ])
    
#%%
wfSPM_rsa.write_graph(graph2use = 'flat')

# # wfSPM.write_graph("workflow_graph.dot", graph2use='colored', format='png', simple_form=True)
# # wfSPM.write_graph(graph2use='orig', dotfilename='./graph_orig.dot')
# %matplotlib inline
# from IPython.display import Image
# %matplotlib qt
# Image(filename = '/home/rj299/project/mdm_analysis/work/l1spm/graph.png')
    
#%% run
wfSPM_rsa.run('MultiProc', plugin_args={'n_procs': 4})
#wfSPM_rsa.run(plugin='Linear', plugin_args={'n_procs': 1})    
