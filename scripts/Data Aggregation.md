---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import sys
!{sys.prefix}/bin/pip install -e energetic_costs/
#../
#Restart the kernel after installing
os._exit(00)
```

#### Libraries loading

```python
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from matplotlib import pyplot as plt
%matplotlib inline

import os,pickle
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from nilearn import datasets, input_data
import enigmatoolbox
from enigmatoolbox.utils.parcellation import surface_to_parcel,parcel_to_surface

#BrainSmash
from brainsmash.mapgen.base import Base 
from brainsmash.mapgen.eval import base_fit
from brainsmash.mapgen.stats import pearsonr, pairwise_r, nonparp

plt.rcParams['font.sans-serif'] = ['Open Sans']+plt.rcParams['font.sans-serif']
sns.set_context("notebook", font_scale=2.1)
sns.set_style("whitegrid")

import src.functions
%load_ext autoreload
%autoreload 2
```

#### General variables and filepaths

```python
root_dir = '../data'
# BIDS directory downloaded from https://openneuro.org/datasets/ds004513
bids_dir = '../../../fdgquant2016/4eliana2convert2bids/bids2upload/derivatives/energetic-costs'
#change it for bids_dir = 'bidsdir/derivatives/energetic-costs'

conn_metric = 'degree'
dc_type = 'weighted'
pet_metric = 'cmrglc'
atlas_suf = 'mmp'
vol_res = '3mm'
vol_space = 'mni-'+vol_res
dc_z = '_z' 
y_var = pet_metric
x_var = conn_metric+dc_z
xlabel='dFC [Z-score]'
ylabel='CMRglc [umol/(min*100g)]'

gm_thr = 25
fc_res_label = 'cpac_v1.4.0'
dti_res_label = 'mrtrix3_v0.4.2'
pipeline = '_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic1.gm0.compcor1.csf1'
lh_dist_full = np.loadtxt(os.path.join(root_dir,'external','brainSMASH','LeftParcelGeodesicDistmat.txt'))


sel_cm = 'RdBu_r'
gray_c = [0.77,0.77,0.77,1]
extended_cm=np.concatenate((np.array([gray_c]),getattr(plt.cm,sel_cm)(np.arange(0,getattr(plt.cm,sel_cm).N))))
fig_res_dpi = 300
s = 0.1
```

#### Data Loading

```python
total_n_subj = 47
cohorts_metadata_fn = os.path.join(root_dir,f'gx_all-cohorts_data_nsubj-{total_n_subj}_{conn_metric}-{dc_type}.pickle')
if not os.path.exists(cohorts_metadata_fn):
    !wget https://openneuro.org/crn/datasets/ds004513/files/derivatives:energetic-costs:gx_all-cohorts_data_nsubj-47_degree-weighted.pickle -O {cohorts_metadata_fn}
with open(cohorts_metadata_fn, 'rb') as f:
    cohorts_metadata = pickle.load(f)

tum_participants_fn = os.path.join(root_dir,'._participants.tsv')
if not os.path.exists(tum_participants_fn):
    !wget https://openneuro.org/crn/datasets/ds004513/files/participants.tsv -O {tum_participants_fn}
tum_participants_info = pd.read_csv(tum_participants_fn,sep='\t')
vie_participants_info = pd.read_csv(os.path.join(root_dir,'external','Sundar2018','VIE_participants.tsv'),sep='\t')
age_participants_mapping = {**dict(zip(tum_participants_info.participant_id.str.split('-').str[1],tum_participants_info.age)),**dict(zip(vie_participants_info.participant_id.str.split('-').str[1],vie_participants_info.age))}
sex_participants_mapping = {**dict(zip(tum_participants_info.participant_id.str.split('-').str[1],tum_participants_info.sex)),**dict(zip(vie_participants_info.participant_id.str.split('-').str[1],vie_participants_info.sex))}

```

#### Atlas

```python
mmp_atlas_fn = os.path.join('../../..',f'MMP_in_MNI_corr_{vol_res}.nii.gz')
mmp_n = 360
N = '7'
n = '400'
yeo_suf = n+'_'+N+'N'
schaefer_info = pd.read_csv(os.path.join(root_dir,'external','Schaefer2018_'+yeo_suf+'_order.txt'),sep='\t',header=None)
schaefer_info['network'] = schaefer_info[1].str.split('_').str.get(2)
nw_label2id = dict(zip(schaefer_info['network'].unique(),range(1,int(N)+1)))
nw_id2label=dict(zip(range(1,int(N)+1),schaefer_info['network'].unique()))
schaefer_info['network_id'] = schaefer_info['network'].map(nw_label2id)
schaefer_roi2nw = dict(zip(schaefer_info[0].tolist(), schaefer_info['network_id']))
schaefer_info['network_id'] = schaefer_info['network'].map(nw_label2id)
yeo2mmp = enigmatoolbox.utils.parcellation.surface_to_parcel(enigmatoolbox.utils.parcellation.parcel_to_surface(schaefer_info['network_id'].to_numpy(),
                                                                                                                'schaefer_{}_conte69'.format(n)),
                                                             'glasser_360_conte69',red_op='mode')
mmp_dict={}
mmp_dict['roi2network'] = dict(zip(range(1,int(mmp_n)+1),np.vectorize(nw_id2label.get)(yeo2mmp[1:].astype(int)).flatten()))
yeo_colors = np.array(pd.read_csv(getattr(datasets.fetch_atlas_yeo_2011(),'colors_'+N),sep='\s+').iloc[:,2:5]/255)
yeo_nw_colors = {nw_id2label[i+1]: (yeo_colors[i,0],yeo_colors[i,1],yeo_colors[i,2]) for i in range(len(nw_id2label))} 
mmp_dict['nw2color'] = yeo_nw_colors
ignore_yeo_nws = ['Other','None',None,'Limbic']
```

#### Extraction and aggregation from nifti data into pandas dataframe

```python
df_labels = {'all_ind_vox_vals':pd.DataFrame({}),'all_avg_vox_vals':pd.DataFrame({}),'all_avg_vox_vals_with_gx_mask':pd.DataFrame({})}
for df_label in df_labels.keys():
    all_df = pd.DataFrame({})
    if df_label=='all_avg_vox_vals_with_gx_mask':
        gmm_fn = os.path.join(bids_dir,f'sub-all_space-MNI152NLin6ASym_res-{vol_res}_desc-GM_mask.nii.gz')
    for site in list(cohorts_metadata.keys())[:1]:
        for coh in cohorts_metadata[site].keys():
            bids_ses = "ses-open" if cohorts_metadata[site][coh]["session"]=="AUF" else "ses-closed"
            if df_label=='all_avg_vox_vals':
                gmm_fn = os.path.join(bids_dir,f'sub-{site}{coh}_{bids_ses}_space-MNI152NLin6ASym_res-{vol_res}_desc-GM_mask.nii.gz')
            for sid in cohorts_metadata[site][coh]['sids']:
                bids_sub = f'sub-s{sid:03d}'
                bids_base_fn = f'{bids_sub}_{bids_ses}_task-rest_space-MNI152NLin6ASym_res-{vol_res}_desc-'
                sub_dir = os.path.join(bids_dir,bids_sub,bids_ses)
                if df_label=='all_ind_vox_vals':
                    gmm_fn = os.path.join(sub_dir,'anat',f'{bids_base_fn}GM_mask.nii.gz'.replace('_task-rest',''))
                dfc_fn = os.path.join(sub_dir,'func',f'{bids_base_fn}dFC_bold.nii.gz')
                cmrglc_fn = os.path.join(sub_dir,'pet',f'{bids_base_fn}CMRglc_pet.nii.gz')
                vbm_fn = os.path.join(sub_dir,'anat',f'{bids_base_fn}VBM_T1w.nii.gz'.replace('_task-rest',''))
                ecosts_fn = os.path.join(sub_dir,'pet',f'{bids_base_fn}signcosts_pet.nii.gz')                
                df = src.functions.nii2df({'mask':gmm_fn,'roi_id':mmp_atlas_fn, conn_metric:dfc_fn,pet_metric:cmrglc_fn,'energetic_costs':ecosts_fn, 'gm_vbm':vbm_fn,
                                           'other_fields':{'sid':bids_sub.split('-')[1],'session':bids_ses.split('-')[1],'cohort':f'{site}.{coh}'}}
                                         )
                df['vox_id'] = np.arange(df.shape[0])
                df['roi_id'] = df['roi_id'].astype(int)
                df.loc[df['roi_id']>180,'roi_id'] = df.loc[df['roi_id']>180,'roi_id']-20 #MMP righ ROIS start at 200 istead of 181 
                df['nw'] = df['roi_id'].map(mmp_dict['roi2network'])
                df = df[((df[conn_metric]>0) & (~df['nw'].isin(ignore_yeo_nws)) & (~df['nw'].isna()))]
                df[conn_metric+dc_z] = stats.zscore(df[conn_metric])
                df['gm_vbm'+dc_z] = stats.zscore(df['gm_vbm'])
                all_df = pd.concat([all_df,df], ignore_index=True)
                if df_label=='all_ind_vox_vals':
                    if sid not in cohorts_metadata[site][coh]['individual_smash'].keys():
                        cohorts_metadata[site][coh]['individual_smash'][sid] = {}
                    if f'smash_{x_var}-{y_var}' not in cohorts_metadata[site][coh]['individual_smash'][sid].keys():
                        cohorts_metadata[selected_site][coh]['individual_smash'][sid][f'smash_{x_var}-{y_var}'] = src.functions.smash_comp(src.functions.metric2mmp(df,x_var,'roi_id'),metric2mmp(df,y_var,'roi_id'),lh_dist_full,
                                                                                                                                           y_nii_fn=cmrglc_fn,l=5,u=95,n_mad='min',p_uthr=0.05,plot=False)
            if df_label=='all_avg_vox_vals':
                if f'smash_{x_var}-{y_var}' not in cohorts_metadata[site][coh].keys():
                    cohorts_metadata[site][coh][f'smash_{x_var}-{y_var}'] = src.functions.smash_comp(src.functions.metric2mmp(all_df[all_df.cohort==f'{site}.{coh}'].groupby(['nw','vox_id'], as_index=False).median(),x_var,'roi_id'),
                                                                                                     src.functions.metric2mmp(all_df[all_df.cohort==f'{site}.{coh}'].groupby(['nw','vox_id'], as_index=False).median(),y_var,'roi_id'),
                                                                                                     lh_dist_full,l=5,u=95,n_mad='min',p_uthr=0.05,plot=False,y_nii_fn=os.path.join(bids_dir,f'{site}.{coh}_{x_var}-{y_var}.nii.gz'))
            elif ((df_label=='all_avg_vox_vals_with_gx_mask') & (f'{site}.{coh}'!='TUM.exp1')):
                ## similarity between the distribution of energetic costs of signaling in TUM.exp1 and each of the other cohorts
                if 'all' not in cohorts_metadata.keys():
                    cohorts_metadata['all'] = {}
                if f'smash_ecosts_TUM.exp1-{site}.{coh}' not in cohorts_metadata['all'].keys():
                    cohorts_metadata['all'][f'smash_ecosts_TUM.exp1-{site}.{coh}'] = src.functions.smash_comp(src.functions.metric2mmp(all_df[all_df.cohort=='TUM.exp1'].groupby(['nw','vox_id'], as_index=False).median(),'energetic_costs','roi_id'),
                                                                                                              src.functions.metric2mmp(all_df[all_df.cohort==f'{site}.{coh}'].groupby(['nw','vox_id'], as_index=False).median(),'energetic_costs','roi_id'),
                                                                                                              lh_dist_full,l=5,u=95,n_mad='min',p_uthr=0.05,plot=False,y_nii_fn=os.path.join(bids_dir,f'smash_ecosts_TUM.exp1-{site}.{coh}.nii.gz'))
                    

    if df_label=='all_ind_vox_vals':
        all_df.drop('vox_id', axis=1, inplace=True)
        df_labels[df_label] = all_df.copy()
    elif ((df_label=='all_avg_vox_vals') | (df_label=='all_avg_vox_vals_with_gx_mask')):
        df_labels[df_label] = all_df.groupby(['cohort','nw','vox_id'], as_index=False).median()
        df_labels[df_label]['vox_id'] = df_labels[df_label]['vox_id'].astype(int)
        df_labels[df_label]['roi_id'] = df_labels[df_label]['roi_id'].astype(int)
        if df_label=='all_avg_vox_vals':
            all_avg_roi_vals = df_labels[df_label].groupby(['cohort','nw','roi_id'], as_index=False).median()
            all_avg_roi_vals['roi_id'] = all_avg_roi_vals['roi_id'].astype(int)
            all_avg_roi_vals.drop('vox_id', axis=1, inplace=True)
        elif df_label=='all_avg_vox_vals_with_gx_mask':
            if f'smash_{x_var}-{y_var}' not in cohorts_metadata['all'].keys():
                cohorts_metadata['all'][f'smash_{x_var}-{y_var}'] = src.functions.smash_comp(src.functions.metric2mmp(df_labels[df_label].groupby(['vox_id'], as_index=False).median(),x_var,'roi_id'),
                                                                                             src.functions.metric2mmp(df_labels[df_label].groupby(['vox_id'], as_index=False).median(),y_var,'roi_id'),
                                                                                             lh_dist_full,l=5,u=95,n_mad='min',p_uthr=0.05,plot=False,y_nii_fn=os.path.join(bids_dir,f'avg_all_{x_var}-{y_var}.nii.gz'))
            
            
        

```

#### Dataframes saving into csv file

```python
df_labels['all_ind_vox_vals'].to_csv(os.path.join(root_dir,f'individual_all-cohorts_vox_nsubj-{total_n_subj}_{conn_metric}-{dc_type}.csv.zip'))
df_labels['all_avg_vox_vals'].to_csv(os.path.join(root_dir,f'gx_all-cohorts_vox_nsubj-{total_n_subj}_{conn_metric}-{dc_type}.csv'))
all_avg_roi_vals.to_csv(os.path.join(root_dir,f'gx_all-cohorts_roi_nsubj-{total_n_subj}_{conn_metric}-{dc_type}.csv'))
df_labels['all_avg_vox_vals_with_gx_mask'].to_csv(os.path.join(root_dir,f'gx_all-cohorts_vox_gx-mask_nsubj-{total_n_subj}_{conn_metric}-{dc_type}.csv.zip'))
with open(cohorts_metadata_fn, 'wb') as f:
    pickle.dump(cohorts_metadata, f, pickle.HIGHEST_PROTOCOL)

```
