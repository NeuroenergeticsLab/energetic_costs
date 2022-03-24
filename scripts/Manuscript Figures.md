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
    display_name: energy_density_env (ed)
    language: python
    name: ed
---

```python
from matplotlib import pyplot as plt
%matplotlib inline
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import os,glob,pickle,re,sys,json
import pandas as pd
import numpy as np
from scipy import stats

import enigmatoolbox
from enigmatoolbox.utils.parcellation import surface_to_parcel,parcel_to_surface
```

<!-- #region tags=[] -->
#### Variables declaration and filepaths
<!-- #endregion -->

```python
root_dir = '../data'
img_dir = os.path.join(root_dir,'img_files')

thr=0.25
thr_i = "%i" % (thr*100)

fc_res_label = 'cpac_v1.4.0'
dti_res_label = 'mrtrix3_v0.4.2'
pipeline='_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1' if fc_res_label=='cpac_v1.6.1' else '_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic1.gm0.compcor1.csf1'
ref_img = '/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain.nii.gz'
lh_dist_full = np.loadtxt('ext_data/brainsmash/example_data/LeftParcelGeodesicDistmat.txt')


conn_metric = 'degree'#'degree' 'dti' 'alff' 'gmvar' shannon_entropy
dc_type = 'weighted'#'weighted' #binarize
pet_metric = 'cmrglc'


sthr_suf = '' #'_sthr-1'
smooth_suf = '_fwhm-6'
recon_tag='_45min'#'45sfrscanner'#'_45min'
vol_space = 'mni-'+vol_res
recon_label='_1120'
recon_tag='_45min'
qx_t0 = 22 #21 #22
qx_tend = 42 #45 #42
pvc = '_pvc-pveseg'#'_pvc-pveseg'
pvc_suf = '_pvc-pveseg' #'_pvc-pveseg'
calc_z = False # True for DTI
nmad = 2.5 if conn_metric == 'alff' else ''#2 if conn_metric == 'alff' else '' 80
freq_band= '_0.01.0.1' if conn_metric == 'alff' else '' 
z_suff=''
GSR=''# _GSR-75 '_GSR-90'
w_length=20
w_step=10
dyn_wc =''#'_wl'+str(w_length)+'ws'+str(w_step)+'_dyn_STD' #'_wl'+str(w_length)+'ws'+str(w_step)+'_dyn_VAR'
dc_z = '_z' 
y_var = pet_metric
x_var = conn_metric+dc_z
if dyn_wc!='':
    x_label='std(DC_window) [a.u.]' if  dc_z=='' else 'std(DC_window) [Z-score]'
elif conn_metric=='DTI':
    x_label='DTI strength [a.u.]'
elif conn_metric=='alff':
    x_label='ALFF [a.u.]' if  dc_z=='' else 'ALFF [Z-score]'
elif conn_metric=='shannon_entropy':
    x_label='Shannon entropy [bits]' if not len(dc_z) else 'Shannon entropy [Z-score]'
else:
    x_label=''
xlabel='DC [Z-score]' if not len(x_label) else x_label
ylabel='CMRglc [umol/(min*100g)]' if y_var == pet_metric else xlabel
xlabel=xlabel if y_var == pet_metric else 'CMRglc [umol/(min*100g)]'
```

#### Figure 1

```python
save_df = False
load_df = False
plot_signden = True
plot_expansion = True
expresion_log = True
plot_mod_maps = False
voxelwise = True
s = 0.1 if voxelwise else 10
sd_res_roi_df = pd.DataFrame({})
#ylim=(5, 60) if voxelwise else (15,45)
if not load_df:
    all_avg_vox_vals = pd.DataFrame({})
    all_avg_roi_vals = pd.DataFrame({})
    total_n_subj = 0
else:
    total_n_subj = 47
    all_avg_vox_vals = pd.read_csv(os.path.join(root_dir,'fdgquant2016','gx_all-cohorts_vox_nsubj-{}_{}-{}_v1.0.csv'.format(total_n_subj,conn_metric,dc_type)))
    if 'index' in all_avg_vox_vals.columns: all_avg_vox_vals.drop(['index'], axis = 1, inplace=True)
    all_avg_roi_vals = pd.read_csv(os.path.join(root_dir,'fdgquant2016','gx_all-cohorts_roi_nsubj-{}_{}-{}_v1.0.csv'.format(total_n_subj,conn_metric,dc_type))) 
    if 'index' in all_avg_roi_vals.columns: all_avg_roi_vals.drop(['index'], axis = 1, inplace=True)
#list(cohorts_metadata.keys())[:-1] when site ='all' is already included
for site in list(cohorts_metadata.keys()):#[:-1]:#cohorts_metadata.keys():#list(cohorts_metadata.keys())[:1]:#
    for coh in cohorts_metadata[site].keys():#list(cohorts_metadata[site].keys())[:1]:#
        project_id = cohorts_metadata[site][coh]['project_id']
        session = cohorts_metadata[site][coh]['session']
        version = cohorts_metadata[site][coh]['version']
        total_n_subj += cohorts_metadata[site][coh]['n_subj']
        cpac_dir = os.path.join(root_dir,project_id,fc_res_label,session)
        ses_dir = os.path.join(root_dir,project_id,session)
        if not load_df:
            gm_mask_pref = 'mean_'+session+'_segment_seg_1_'+thr_i+'_mni-'+vol_res+'_'+atlas_suf+version
            gx_gm_mask_fn = os.path.join(cpac_dir,gm_mask_pref+'.nii.gz')#_wo_outliers #root_dir inst. cpac_dir
            #!GROUP GM '/RAID1/jupytertmp/fdgquant2016/all_mean_segment_seg_1_25_mni-3mm_mmp_vXsubQmmp.nii.gz'
            if 'gx_gm_mask_fn' not in cohorts_metadata[site][coh].keys():
                cohorts_metadata[site][coh]['gx_gm_mask_fn'] = gx_gm_mask_fn
            if join_mask:  
                gx_gm_mask_fn = gx_gm_mask_fn.replace(session+'/','').replace(session,'joined').replace(v_year,'').replace('_v'+str(n_subj),'_v6')       
                print('joint mask')
            gx_gm_masker = input_data.NiftiMasker(mask_img=gx_gm_mask_fn)
            avg_ki_file = os.path.join(ses_dir,'all_'+session+'_1.3.12.2_itr-4_trimmed-upsampled-scale-2'+recon_tag+'_mcf'+smooth_suf+'_quant-cmrglc_acq-'+str(qx_t0)+str(qx_tend)+'min'+pvc+'_mni-'+vol_res+version+'.nii.gz') if project_id in ['fdgquant2016','dcmet_hc_wien'] else os.path.join(ses_dir,'all_'+session+'_rec-'+recon_tag+'_acq-0-45min_pet_mcf_regout-model_acq-'+str(qx_t0)+str(qx_tend)+'min'+pvc+'_mni-'+vol_res+version+'.nii.gz')
            avg_ki = gx_gm_masker.fit_transform(avg_ki_file)
            
            yeo_rois_fn = os.path.join(cpac_dir,gm_mask_pref+'_rois.nii.gz')
            yeo_rois = gx_gm_masker.fit_transform(yeo_rois_fn).flatten()
            yeo_rois[yeo_rois>180]=yeo_rois[yeo_rois>180]-20
            if 'mmp_rois' not in cohorts_metadata[site][coh].keys():
                cohorts_metadata[site][coh]['mmp_rois'] = yeo_rois
            yeo_labels=np.vectorize(atlas_dict['roi2network'].get)(yeo_rois.astype(int)).flatten()
            consmod_labels = np.vectorize(mmp2consmod_dict.get)(yeo_rois).flatten()    
        #    kcore_labels = np.vectorize(kcore_dict.get)(yeo_rois).flatten()
            if conn_metric=='degree':
                avg_met_file = os.path.join(cpac_dir,'all_'+session+'_'+conn_metric+'_centrality_{}_mnireg-{}_gm-{}{}'.format(dc_type,vol_res,thr_i,calc_dc_bct)+freq_band+GSR+dyn_wc+sthr_suf+smooth_suf+z_suff+version+'.nii.gz')# 
            if conn_metric=='shannon_entropy':
                avg_met_file = os.path.join(cpac_dir,'all_'+session+'_'+conn_metric+'_bits{}_mni-{}{}.nii.gz'.format(smooth_suf,vol_res,version))
                if not os.path.exists(avg_met_file):
                    merge_nii(avg_met_file,cpac_dir+'/resting_preproc_sub-',
                              '_ses-1/afni_centrality_0_degree/_scan_rest/{}/frequency_filter/afni_centrality/*_shannon_entropy_bits_fwhm-6_mni-{}.nii.gz'.format(pipeline,vol_res),
                              [cohorts_metadata[site][coh]['sub_pref'] % sid for sid in cohorts_metadata[site][coh]['sids']])
            elif conn_metric=='gmvar': 
                avg_met_file = os.path.join(cpac_dir,'all_{}_{}_mni-{}_gm-{}{}{}.nii.gz'.format(session,conn_metric,vol_res,thr_i,smooth_suf,version))#_mean_norm
            elif conn_metric=='alff':
                avg_met_file = os.path.join(cpac_dir,'all_'+session+'_'+conn_metric+'_mnireg-{}_gm-{}'.format(vol_res,thr_i)+z_suff+version+freq_band+'.nii.gz')
            met_val = gx_gm_masker.fit_transform(avg_met_file)
            met_z_val = np.array([])
            for dcv in met_val: met_z_val = stats.zscore(dcv)[np.newaxis,:] if not met_z_val.size else np.concatenate((met_z_val,stats.zscore(dcv)[np.newaxis,:]),axis=0)
            met_val = np.nanmedian(met_val,axis=0)
            met_z_val = np.nanmedian(met_z_val,axis=0)
            avg_ki = np.nanmedian(avg_ki,axis=0)
            all_ineff_file = os.path.join(ses_dir,'all_'+session+'_signden_1.3.12.2_itr-4_trimmed-upsampled-scale-2'+recon_tag+'_mcf'+smooth_suf+'_quant-cmrglc_acq-'+str(qx_t0)+str(qx_tend)+'min'+pvc+'_mni-'+vol_res+version+sthr_suf+'.nii.gz')
            if not os.path.exists(all_ineff_file):
                avg_ineff = np.array([])
                for sid in cohorts_metadata[site][coh]['sids']:
                    sess_id = (cohorts_metadata[site][coh]['sub_pref']+'-'+session) % sid
                    sid_dir =  os.path.join(root_dir,project_id,session,'pet',sess_id,'niftypet'+recon_label)
                    avg_ineff_tmp = gx_gm_masker.fit_transform(os.path.join(sid_dir,'signden{}-{}{}_mcf{}_quant-{}_acq-{}{}min{}_gm-{}{}.nii.gz'.format(x_var,y_var,recon_tag,smooth_suf,pet_metric,qx_t0,qx_tend,pvc_suf,thr_i,sthr_suf))).flatten()
                    avg_ineff = avg_ineff_tmp[np.newaxis,:] if not avg_ineff.size else np.concatenate((avg_ineff,avg_ineff_tmp[np.newaxis,:]),axis=0)
            else:
                avg_ineff = gx_gm_masker.fit_transform(all_ineff_file)                
            avg_ineff[avg_ineff==0]=np.nan
            avg_ineff = np.nanmean(avg_ineff,axis=0)
            n_vox = avg_ki.shape[0]
            if nmad:
                non_none_index = ((~np.in1d(yeo_labels,['Other','None',None,'Limbic'])) & (met_val>0) & (~np.isnan(met_val)))
                vals_med = np.nanmedian(met_val[non_none_index])
                vals_mad = stats.median_absolute_deviation(met_val[non_none_index],nan_policy='omit')
                vals_max = np.nanmax(met_val[non_none_index])
                print((vals_med+nmad*vals_mad))
                print((vals_med-nmad*vals_mad))
                met_z_val[(met_val>(vals_med+nmad*vals_mad)) | (met_val<(vals_med-nmad*vals_mad))] = np.nan
                met_val[(met_val>(vals_med+nmad*vals_mad)) | (met_val<(vals_med-nmad*vals_mad))] = np.nan      
                #if conn_metric=='alff':
                #    print('thresholding by +/-{} MAD'.format(nmad))
                #    met_val[(met_val>(vals_med+nmad*vals_mad)) | (met_val<(vals_med-nmad*vals_mad))] = np.nan #(met_val>(vals_med+nmad*vals_mad)) | 
                #else:
                #    print('thresholding by -{} MAD'.format(nmad))
                #    met_val[(met_val>(vals_med+nmad*vals_mad)) | (met_val<(vals_med-nmad*vals_mad))] = np.nan #(met_val>(vals_med+nmad*vals_mad)) | 
            non_none_index = ((~np.in1d(yeo_labels,['Other','None',None,'Limbic'])) & (met_val>0) & (~np.isnan(met_val)))
            if 'non_none_index'.format(x_var,y_var) not in cohorts_metadata[site][coh].keys():
                cohorts_metadata[site][coh]['non_none_index'] = non_none_index    
            avg_vox_vals =  pd.DataFrame({conn_metric:met_val[non_none_index],
                                          conn_metric+'_z':met_z_val[non_none_index],
                                          pet_metric:avg_ki[non_none_index],
                                          'nw':yeo_labels[non_none_index],
                                          'roi_id':yeo_rois[non_none_index],
                                          'module':consmod_labels[non_none_index],
                                          'signal_density':avg_ineff[non_none_index],
                                          'session':session,
                                          'cohort':'{}.{}'.format(site,coh)#,
#                                          'vox_id':np.arange(len(non_none_index))
                                         })
            avg_vox_vals['expansion'] = avg_vox_vals['roi_id'].map(external_datasets['expansion']['mmp_map'])
            avg_vox_vals['expansion_type'] = avg_vox_vals['roi_id'].map(external_datasets['expansion']['categories'])
            if x_var==pet_metric: avg_vox_vals['signal_density'] = -1*avg_vox_vals['signal_density']
            #avg_vox_vals['expansion_type'] = 0
            #avg_vox_vals.loc[avg_vox_vals['expansion']<avg_vox_vals['expansion'].quantile(0.25),'expansion_type'] = -1
            #avg_vox_vals.loc[avg_vox_vals['expansion']>avg_vox_vals['expansion'].quantile(0.75),'expansion_type'] = 1
            
            #if plot_se: avg_vox_vals['shannon_entropy'] = np.nanmedian(gx_gm_masker.fit_transform(os.path.join(cpac_dir,'all_'+session+'_shannon_entropy{}_mni-{}{}.nii.gz'.format(smooth_suf,vol_res,version))),axis=0)[non_none_index]
            #if plot_exp:
                #%store -r mmp2exp
                #avg_vox_vals['expansion'] = avg_vox_vals['roi_id'].map(mmp2exp)
                #!avg_vox_vals['expansion'] = gx_gm_masker.fit_transform('ext_data/Wei2019/chimp2humanF.smoothed15.vol-3mm.nii.gz').flatten()[non_none_index]        
        
            if (conn_metric=='dti'): avg_vox_vals.loc[avg_vox_vals[conn_metric]>3,conn_metric] = np.nan
            if (conn_metric=='gmvar'): 
                vals_med = avg_vox_vals[conn_metric].median()
                vals_std = avg_vox_vals[conn_metric].std()
                avg_vox_vals.loc[(avg_vox_vals[conn_metric]>(vals_med+.5*vals_std)) | (avg_vox_vals[conn_metric]<(vals_med-.5*vals_std)),conn_metric] = np.nan
            elif calc_z:
                avg_vox_vals.loc[~np.isnan(avg_vox_vals[conn_metric]),conn_metric] = stats.zscore(avg_vox_vals.loc[~np.isnan(avg_vox_vals[conn_metric]),conn_metric])
        
            all_avg_vox_vals = pd.concat([all_avg_vox_vals,avg_vox_vals], ignore_index=True)
            
            avg_roi_vals = avg_vox_vals.groupby(['roi_id'], as_index=False).mean()
            avg_roi_vals['nvox_per_roi']=avg_vox_vals.groupby(['roi_id'], as_index=False).count()[pet_metric].to_numpy()
            avg_roi_vals['roi_id'] = avg_roi_vals['roi_id'].astype(int)
            avg_roi_vals = avg_roi_vals.merge(comm_df, on = 'roi_id', how = 'left')
            avg_roi_vals=avg_roi_vals[avg_roi_vals['roi_id']!=0]
            avg_roi_vals['nw'] = avg_roi_vals['roi_id'].map(atlas_dict['roi2network'])
            avg_roi_vals['module'] = avg_roi_vals['roi_id'].map(mmp2consmod_dict)
            avg_roi_vals['session'] = session
            avg_roi_vals['cohort'] = '{}.{}'.format(site,coh)
            all_avg_roi_vals = pd.concat([all_avg_roi_vals,avg_roi_vals], ignore_index=True)
        
            if conn_metric!='dti':    
                avg_vox_vals.loc[avg_vox_vals['nw'] == 'None','nw'] = np.nan
        else:
            avg_vox_vals = all_avg_vox_vals[all_avg_vox_vals.cohort=='{}.{}'.format(site,coh)].copy()
            avg_roi_vals = all_avg_roi_vals[all_avg_roi_vals.cohort=='{}.{}'.format(site,coh)].copy()
            
    
        
        if ('smash_{}-{}'.format(x_var,y_var) not in cohorts_metadata[site][coh].keys()):
            cohorts_metadata[site][coh]['smash_{}-{}'.format(x_var,y_var)] = smash_comp(metric2mmp(avg_vox_vals,x_var,'roi_id'),metric2mmp(avg_vox_vals,y_var,'roi_id'),
                                                              lh_dist_full,l=5,u=95,n_mad='min',p_uthr=0.05,plot=False,
                                                              y_nii_fn=remove_ext(avg_ki_file.replace('all','mean')) if y_var==pet_metric else remove_ext(avg_met_file.replace('all','mean')))
#DEL        elif cohorts_metadata[site][coh]['smash_{}-{}'.format(x_var,y_var)]==None:
#DEL            cohorts_metadata[site][coh]['smash_{}-{}'.format(x_var,y_var)] = smash_comp(metric2mmp(avg_vox_vals,x_var,'roi_id'),metric2mmp(avg_vox_vals,y_var,'roi_id'),
#DEL                                                              lh_dist_full,l=5,u=95,n_mad='min',p_uthr=0.05,plot=False,
#DEL                                                              y_nii_fn=remove_ext(avg_ki_file.replace('all','mean')) if y_var==pet_metric else remove_ext(avg_met_file.replace('all','mean')))
            
        r_param,p_param=stats.pearsonr(avg_vox_vals.loc[avg_vox_vals[conn_metric].notnull(),x_var],avg_vox_vals.loc[avg_vox_vals[conn_metric].notnull(),y_var])
        if len(cohorts_metadata[site][coh]['smash_{}-{}'.format(x_var,y_var)])>0:
            p_np = nonparp(r_param, cohorts_metadata[site][coh]['smash_{}-{}'.format(x_var,y_var)])
            p_np = p_np if p_np>0 else 0.00001
#        g = plot_joint(avg_vox_vals[y_var],avg_vox_vals[conn_metric],s=s,robust=False,kdeplot=False,truncate=True,xlim0=False,
#                   ylim=(3.2, 5.7),#xlim=(0, 50),
#                   y_label=ylabel,x_label=xlabel,return_plot_var=True,p_smash=p_np,thr_data=3,plot_log=True)
            g = plot_joint(avg_vox_vals[x_var],avg_vox_vals[y_var],s=s,robust=False,kdeplot=False,truncate=True,xlim0=False,
                       #ylim=(3.2, 5.7),#xlim=(0, 50),
                           y_label=ylabel,x_label=xlabel,return_plot_var=True,p_smash=p_np)
        else:
            g = plot_joint(avg_vox_vals[x_var],avg_vox_vals[y_var],s=s,robust=False,kdeplot=False,truncate=True,xlim0=False,y_label=ylabel,x_label=xlabel,return_plot_var=True)
        plt.suptitle('{}.{}'.format(site,coh))
        if plot_signden:
            sns.scatterplot(x=x_var, y=y_var, hue='signal_density', #alpha=0.75,
                data=avg_vox_vals,linewidth=0,s=1.5,legend=False,palette=sel_cm,
                vmin=avg_vox_vals.signal_density.quantile(0.25),vmax=avg_vox_vals.signal_density.quantile(0.75),ax=g.ax_joint)
            plot_surf(metric2mmp(avg_vox_vals,'signal_density','roi_id'), 
                      os.path.join(root_dir,project_id,session,'pet','{}.{}_signden'.format(site,coh)),
                      cmap=ListedColormap(extended_cm),colorbar=True,vlow=5,vhigh=95,fig_title='Signal density {}.{}'.format(site,coh)) 
        if plot_mod_maps:
            plot_surf(metric2mmp(avg_vox_vals,x_var,'roi_id'), 
                      os.path.join(root_dir,project_id,session,'pet','{}.{}_{}'.format(site,coh,x_var)),
                      cmap=ListedColormap(np.concatenate((np.array([[0.5,0.5,0.5,1.0]]),getattr(plt.cm,'viridis')(np.arange(0,getattr(plt.cm,'viridis').N))))),
                      colorbar=True,vlow=10,vhigh=90,fig_title='{} {}.{}'.format(x_var,site,coh))
            plot_surf(metric2mmp(avg_vox_vals,y_var,'roi_id'), 
                      os.path.join(root_dir,project_id,session,'pet','{}.{}_{}'.format(site,coh,y_var)),
                      cmap=ListedColormap(np.concatenate((np.array([[0.5,0.5,0.5,1.0]]),getattr(plt.cm,'cividis')(np.arange(0,getattr(plt.cm,'cividis').N))))),
                      colorbar=True,vlow=10,vhigh=90,fig_title='{} {}.{}'.format(y_var,site,coh))
            
#!        plt.figure(figsize=(2,5))
#!        plot_rnd_dist(cohorts_metadata[site][coh]['smash'],r_param,p_np,plt.gca(),xlabel=conn_metric.upper(),print_text=False)
#        gx_avg_roi_vals = avg_vox_vals.groupby(['roi_id'], as_index=False).mean()
#        gx_avg_roi_vals['residuals'] = pg.linear_regression(gx_avg_roi_vals[x_var],gx_avg_roi_vals[y_var],coef_only=False,remove_na=True,as_dataframe=False)['residuals']
#        gx_avg_roi_vals['expansion'] = gx_avg_roi_vals['roi_id'].map(external_datasets['expansion']['mmp_map'])
        plt.figure(figsize=(3,3))
#        sns.scatterplot(x=x_var,y='residuals',hue='expansion',data=gx_avg_roi_vals,palette='Spectral_r',legend=False,s=10)#color=(0.6,0.6,0.6))#, s=3*s)
        #
        avg_vox_vals['residual'] = pg.linear_regression(avg_vox_vals[x_var],avg_vox_vals[y_var],coef_only=False,remove_na=True,as_dataframe=False)['residuals']
        sns.scatterplot(x_var,'residual',data=avg_vox_vals,s=3*s,legend=False,hue='residual', palette=sel_cm,
                        vmin=avg_vox_vals.residual.quantile(0.25),vmax=avg_vox_vals.residual.quantile(0.75))#color=(0.6,0.6,0.6))#,hue='expansion_type', palette='Spectral_r'
        sd_res_roi_df = pd.concat([sd_res_roi_df,avg_vox_vals.groupby(['cohort','roi_id'], as_index=False).mean()[['cohort','roi_id','residual',x_var]]], ignore_index=True) 
        #
        plt.gca().set_xlabel(xlabel)
        plt.gca().set_ylabel('residual')
        if plot_expansion:
            exp_thr = np.log(external_datasets['expansion']['data'][:180]) if expresion_log else external_datasets['expansion']['data'][:180]
            #exp_thr[external_datasets['expansion']['data'][:180]<2.05]=np.min(exp_thr)
            sd_180rois = metric2mmp(avg_vox_vals,'signal_density','roi_id')
            if 'smash_sd_{}-{}'.format(x_var,y_var) not in cohorts_metadata[site][coh].keys():
                cohorts_metadata[site][coh]['smash_sd_{}-{}'.format(x_var,y_var)] = smash_comp(sd_180rois,exp_thr,lh_dist_full,y_nii_fn=os.path.join(img_dir,'expansion_wei2019.png'),l=5,u=95,n_mad=3,
                                                                                               xlabel='Signal density\n[umol/(min*100g)]' if y_var==pet_metric else 'Signal density', ylabel='Brain expansion',
                                                                                               p_uthr=1,plot=True,cmap=ListedColormap(extended_cm),print_text=False,plot_rnd=False,plot_surface=False)
            else:
                smash_comp(sd_180rois,exp_thr,lh_dist_full,y_nii_fn=os.path.join(img_dir,'expansion_wei2019.png'),l=5,u=95,n_mad='min',
                           xlabel='Signal density\n[umol/(min*100g)]' if y_var==pet_metric else 'Signal density', ylabel='Brain expansion',
                           p_uthr=1,plot=True,cmap=ListedColormap(extended_cm),print_text=True,plot_rnd=True,plot_surface=True,
                           x_surr_corrs=cohorts_metadata[site][coh]['smash_sd_{}-{}'.format(x_var,y_var)])
                #r_sd_param,p_param=stats.pearsonr(sd_180rois,exp_thr)
                #p_sd_np = nonparp(r_sd_param, cohorts_metadata[site][coh]['smash_sd_{}-{}'.format(x_var,y_var)])
                #p_sd_np = p_sd_np if p_sd_np>0 else 0.00001
                #plot_joint(sd_180rois,exp_thr,s=28,p_smash=p_sd_np,
                #           x_label='Signal density\n[umol/(min*100g)]' if y_var==pet_metric else 'Signal density', y_label='Brain expansion')           

sd_res_roi_df['expansion'] = sd_res_roi_df['roi_id'].map(external_datasets['expansion']['mmp_map'])
plt.figure(figsize=(3,3))
sns.scatterplot(x=x_var,y='residual',hue='expansion',data=sd_res_roi_df[sd_res_roi_df.cohort!='TUM.b'].groupby(['roi_id'], as_index=False).mean(),palette=sel_cm,legend=False,s=10)#color=(0.6,0.6,0.6))#, s=3*s)
plt.gca().set_xlabel(xlabel)
plt.gca().set_ylabel('residual')        
if save_df:
    all_avg_roi_vals.to_csv(os.path.join(root_dir,'fdgquant2016','gx_all-cohorts_roi_nsubj-{}_{}-{}_v1.0.csv'.format(total_n_subj,conn_metric,dc_type)),index=False)
    all_avg_vox_vals.to_csv(os.path.join(root_dir,'fdgquant2016','gx_all-cohorts_vox_nsubj-{}_{}-{}_v1.0.csv'.format(total_n_subj,conn_metric,dc_type)),index=False)

```
