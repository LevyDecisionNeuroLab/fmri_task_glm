#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:29:06 2019

@author: Or Duek
2nd level analysis using FSL output
"""

#%%
#%% Load packages
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu, io as nio



import os
#%% Set variables
# set number of contrasts (cope)
cope_list = ['1','2','3']
# setting working directory (same as first level)
work_dir = '/home/oad4/scratch60/kpe_work/'
# set input directory (where original files are)
mask_dir = '/home/oad4/scratch60/kpe_fsl/'

#%% Now run second level
workflow2nd = pe.Workflow(name="2nd_level", base_dir=work_dir)

copeInput = pe.Node(niu.IdentityInterface(
        fields = ['cope']),
        name = 'copeInput')
        
copeInput.iterables= [('cope', cope_list)]


#inputnode = pe.Node(niu.IdentityInterface(
#    fields=['group_mask', 'in_copes', 'in_varcopes']),
#    name='inputnode')

#num_copes = 3

# SelectFiles - to grab the data (alternativ to DataGrabber)
templates = {'in_copes': work_dir + 'modelfit/_subject_id_*/modelestimate/mapflow/_modelestimate0/results/cope{cope}.nii.gz',
             'mask': mask_dir + 'derivatives/fmriprep/sub-*/ses-1/func/sub-*_ses-1_task-Memory_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz',
             'in_varcopes': work_dir + 'modelfit/_subject_id_*/modelestimate/mapflow/_modelestimate0/results/varcope{cope}.nii.gz',
             }
selectCopes = pe.Node(nio.SelectFiles(templates,
                               base_directory=work_dir),
                   name="selectCopes")

#%%

copemerge    = pe.Node(interface=fsl.Merge(dimension='t'),
                          name="copemerge")

varcopemerge = pe.Node(interface=fsl.Merge(dimension='t'),
                       name="varcopemerge")

maskemerge = pe.Node(interface=fsl.Merge(dimension='t'),
                       name="maskemerge")
#copeImages = glob.glob('/media/Data/work/firstLevelKPE/_subject_id_*/feat_fit/run0.feat/stats/cope1.nii.gz')
#copemerge.inputs.in_files = copeImages



# Configure FSL 2nd level analysis
l2_model = pe.Node(fsl.L2Model(), name='l2_model')

flameo_ols = pe.Node(fsl.FLAMEO(run_mode='ols'), name='flameo_ols')
def _len(inlist):
    print (len(inlist))
    return len(inlist)
### use randomize
rand = pe.Node(fsl.Randomise(),
                            name = "randomize") 


rand.inputs.mask = '/home/oad4/scratch60/kpe_fsl/derivatives/fmriprep/sub-1369/ses-1/func/sub-1369_ses-1_task-Memory_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz' # group mask file (was created earlier)
rand.inputs.one_sample_group_mean = True
rand.inputs.tfce = True
rand.inputs.vox_p_values = True
rand.inputs.num_perm = 5000
# Thresholding - FDR ################################################
# Calculate pvalues with ztop
fdr_ztop = pe.Node(fsl.ImageMaths(op_string='-ztop', suffix='_pval'),
                   name='fdr_ztop')
# Find FDR threshold: fdr -i zstat1_pval -m <group_mask> -q 0.05
# fdr_th = <write Nipype interface for fdr>
# Apply threshold:
# fslmaths zstat1_pval -mul -1 -add 1 -thr <fdr_th> -mas <group_mask> \
#     zstat1_thresh_vox_fdr_pstat1

# Thresholding - FWE ################################################
# smoothest -r %s -d %i -m %s
# ptoz 0.05 -g %f
# fslmaths %s -thr %s zstat1_thresh

# Thresholding - Cluster ############################################
# cluster -i %s -c %s -t 3.2 -p 0.05 -d %s --volume=%s  \
#     --othresh=thresh_cluster_fwe_zstat1 --connectivity=26 --mm

workflow2nd.connect([
    (copeInput, selectCopes, [('cope', 'cope')]),
    (selectCopes, copemerge, [('in_copes','in_files')]),
    (selectCopes, varcopemerge, [('in_varcopes','in_files')]),
    (selectCopes, maskemerge, [('mask','in_files')]),
    (selectCopes, l2_model, [(('in_copes', _len), 'num_copes')]),
    (copemerge, flameo_ols, [('merged_file', 'cope_file')]),
    (varcopemerge, flameo_ols, [('merged_file', 'var_cope_file')]),
    (maskemerge, flameo_ols, [('merged_file', 'mask_file')]),
    (l2_model, flameo_ols, [('design_mat', 'design_file'),
                            ('design_con', 't_con_file'),
                            ('design_grp', 'cov_split_file')]),
    (copemerge, rand, [('merged_file','in_file')]),
  #  (maskemerge, rand, [('merged_file','mask')]),
    (l2_model, rand, [('design_con','tcon'), ('design_mat','design_mat')]),
    (maskemerge, fdr_ztop, [('merged_file','mask_file')]),
    (flameo_ols, fdr_ztop, [('zstats','in_file')]),
])
#%%
workflow2nd.run()


