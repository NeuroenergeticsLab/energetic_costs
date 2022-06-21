import os,glob
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

import matplotlib.colors as clrs
from matplotlib import colorbar
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

from nilearn import input_data

from brainsmash.mapgen.base import Base 
from brainsmash.mapgen.eval import base_fit
from brainsmash.mapgen.stats import pearsonr, pairwise_r, nonparp

from wbplot import pscalar
from wbplot import constants
from IPython.display import Image
import matplotlib.image as mpimg

def plot_joint(x,y,s=0,x_label='',y_label='', robust=False,xlim=None,ylim=None,return_plot_var=False,p_smash=None,kdeplot=False,truncate=True,xlim0=False,thr_data=False,plot_log=False):
    if thr_data:
        x[pd.isna(x)]=x.min()#0
        y[pd.isna(y)]=y.min()#0
        x = x.astype(float)
        y = y.astype(float)
        if not isinstance(thr_data, str):
            x_mad = stats.median_absolute_deviation(x[x>x.min()],nan_policy='omit') #[x>0]
            x_med = np.nanmedian(x[x>x.min()])
            y_mad = stats.median_absolute_deviation(y[y>y.min()],nan_policy='omit') #[y>0]
            y_med = np.nanmedian(y[y>y.min()])
            valid_ind = ((x<(x_med+(thr_data*x_mad))) & (x>(x_med-(thr_data*x_mad))) & (y<(y_med+(thr_data*y_mad))) & (y>(y_med-(thr_data*y_mad))) & (x>x.min()) & (y>y.min())) #(x>0) & (y>0) & 
        else:
            valid_ind = ((x>np.nanmin(x)) & (y>np.nanmin(y)))
        if plot_log:
            g = sns.jointplot(np.log(x[valid_ind]), np.log(y[valid_ind]), kind="reg",dropna=True,color='k', space=0,joint_kws={'robust':robust},xlim=xlim,ylim=ylim,truncate=truncate)#, stat_func=r,ylim=ylim
        else:
            g = sns.jointplot(x[valid_ind], y[valid_ind], kind="reg",dropna=True,color='k', space=0,joint_kws={'robust':robust},xlim=xlim,ylim=ylim,truncate=truncate)#, stat_func=r,ylim=ylim
    else:
        g = sns.jointplot(x, y, kind="reg",dropna=True,color='k', space=0,joint_kws={'robust':robust},xlim=xlim,ylim=ylim,truncate=truncate)#, stat_func=r,ylim=ylim
    if xlim0: g.ax_joint.set_xlim((0,g.ax_joint.get_xlim()[1]))
    if s>0:
        g.plot_joint(plt.scatter, s=s,color=(0.6,0.6,0.6))
    else:
        g.plot_joint(plt.scatter,color=(0.6,0.6,0.6))
    if x_label: g.ax_joint.set_xlabel(x_label)
    if y_label: g.ax_joint.set_ylabel(y_label)
    g.ax_joint.collections[0].set_alpha(0)
    if kdeplot: g.plot_joint(sns.kdeplot, color="k", zorder=0, levels=6)
    #sns.regplot(x, y,scatter=False,truncate=False)
    if plot_log:
        r,p = stats.pearsonr(np.log(x[valid_ind])[~(np.log(y[valid_ind]).isna())],np.log(y[valid_ind])[~(np.log(y[valid_ind]).isna())])
    else:
        try:
            r,p = stats.pearsonr(x, y)
        except:
            r,p = stats.pearsonr(x[(~np.isnan(x)) & (~np.isnan(y))], y[(~np.isnan(x)) & (~np.isnan(y))])
    p_text = 'p={0:.3f}'.format(p) if p>=0.001 else 'p<0.001' if not p_smash else r'$p_{smash}$=%0.3f' % p_smash  if p_smash>=0.001 else r'$p_{smash}$<0.001' #'$p_{smash}$ = {0:.3f}'.format(p_smash) if p_smash>=0.001 else '$p_{smash}$<0.001'
    g.ax_joint.text(g.ax_joint.get_xlim()[0]+0.02*g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1]-0.02*(g.ax_joint.get_ylim()[1]-g.ax_joint.get_ylim()[0]), 'r={0:.2f}, {1:s}'.format(r,p_text), ha='left',va='top', color='k')
    if return_plot_var:
        return g

def metric2mmp(df,sel_met,roi_id,median=True,hemi='L',calc_log=False):
    avg_vox_vals_mmp = df.copy() #slope_per_roi
    if median:
        avg_vox_vals_mmp = avg_vox_vals_mmp.groupby(roi_id, as_index=False).mean()
    else:
        agg_mode_text = {sel_met: stats.mode}
        avg_vox_vals_mmp = avg_vox_vals_mmp.groupby(roi_id, as_index=False).agg(agg_mode_text)
        avg_vox_vals_mmp[sel_met] = avg_vox_vals_mmp[sel_met].str[0].str[0]
    last_roi = 181 if hemi=='L' else 361
    sel_rois = ((avg_vox_vals_mmp[roi_id]>0) & (avg_vox_vals_mmp[roi_id]<181)) if hemi=='L' else (avg_vox_vals_mmp[roi_id]>180)   
    avg_vox_vals_mmp = avg_vox_vals_mmp.loc[sel_rois]
    missing_rois = list(avg_vox_vals_mmp[roi_id].unique().astype(int))
    last_roi_missing=True if (missing_rois[-1]!=last_roi) else False
    missing_rois = sorted(set(range(1, last_roi)) - set(missing_rois)) if hemi=='L' else sorted(set(range(181, last_roi)) - set(missing_rois))
    if last_roi_missing: missing_rois+=[180]
    print('Missing ROIs: {}'.format(missing_rois))
    min_val = avg_vox_vals_mmp[sel_met].min()
    min_val = min_val-1 if min_val <0 else 0 #
    for rid in missing_rois:
        avg_vox_vals_mmp = avg_vox_vals_mmp.append({roi_id:rid,sel_met:min_val}, ignore_index=True)
    avg_vox_vals_mmp = avg_vox_vals_mmp.sort_values(by=roi_id)
    mmp_sel_met= avg_vox_vals_mmp[sel_met].to_numpy()
    if calc_log: mmp_sel_met = np.log(mmp_sel_met)
    mmp_sel_met[np.isnan(avg_vox_vals_mmp[sel_met].to_numpy())]=min_val if not calc_log else np.log(min_val)
    return mmp_sel_met

def valid_data_index(x,y,n_mad=2):
    x[pd.isna(x)]=x.min()
    y[pd.isna(y)]=y.min()
    x = x.astype(float)
    y = y.astype(float)
    if not isinstance(n_mad, str):
        x_mad = stats.median_absolute_deviation(x[x>x.min()],nan_policy='omit')
        x_med = np.nanmedian(x[x>x.min()])
        y_mad = stats.median_absolute_deviation(y[y>y.min()],nan_policy='omit')
        y_med = np.nanmedian(y[y>y.min()])
        valid_ind = ((x<(x_med+(n_mad*x_mad))) & (x>(x_med-(n_mad*x_mad))) & (y<(y_med+(n_mad*y_mad))) & (y>(y_med-(n_mad*y_mad))) & (x>x.min()) & (y>y.min()))
    else:
        valid_ind = ((x>np.nanmin(x)) & (y>np.nanmin(y)))
    return valid_ind

def smash_comp(x,y,distmat,y_nii_fn='',xlabel='x',ylabel='y',cmap='summer',n_mad=2,rnd_method='smash',l=5,u=95,p_uthr=0.06,colorbar=True,xlim=None,ylim=None,p_xlim=[-0.5,0.5],plot=True,print_text=False,plot_rnd=True,plot_surface=True,x_surr_corrs=None):
    valid_ind = valid_data_index(x,y,n_mad=n_mad)
    test_r,test_p = stats.pearsonr(x[valid_ind], y[valid_ind])
    if test_p<p_uthr:
        dist = distmat[valid_ind,:]
        dist = dist[:,valid_ind]
        if (x_surr_corrs is None):
            if rnd_method=='smash':
                x_gen = Base(x[valid_ind], dist)  # note: can pass numpy arrays as well as filenames
                x_surr_maps = x_gen(n=1000)           
            else:
                x_surr_maps = np.array([np.random.permutation(x[valid_ind]) for _ in range(1000)])
            x_surr_corrs = pearsonr(y[valid_ind], x_surr_maps).flatten()
            return_x_surr_corrs = True
        else:
            return_x_surr_corrs = False
        p_np = nonparp(test_r, x_surr_corrs)
        if plot:
            if p_np<0.08:
                plot_joint(x[valid_ind],y[valid_ind],s=28,robust=False,x_label=xlabel,y_label=ylabel,xlim=xlim,ylim=ylim,p_smash=p_np)
            else:
                plot_joint(x[valid_ind],y[valid_ind],s=28,robust=False,x_label=xlabel,y_label=ylabel,xlim=xlim,ylim=ylim)
            if plot_rnd:
                plt.figure(figsize=(2,5))
                plot_rnd_dist(x_surr_corrs,test_r,p_np,plt.gca(),xlabel=xlabel,ylabel=ylabel,xlim=p_xlim,print_text=print_text)
            if plot_surface: plot_surf(y,remove_ext(y_nii_fn)+'_LH_ROIwise',vlow=l,vhigh=u,cmap=cmap,colorbar=colorbar)
        if return_x_surr_corrs:
            return x_surr_corrs
    else:
        return np.array([])

def vrange(x,l=5,u=95):
    return (np.nanpercentile(x, l), np.nanpercentile(x,u))

def plot_rnd_dist(surr_corrs,r_param,p_non_param,ax,xlabel='',ylabel='',xlim=[-0.5,0.5],print_text=True):
    sns.kdeplot(surr_corrs,shade=True,color=(0.8, 0.8, 0.8),ax=ax) 
    ax.axvline(r_param, 0, 0.95, color='r', linestyle='dashed', lw=1.25)
    # make the plot nicer...
    ax.set_xticks(np.arange(xlim[0], xlim[1]+0.1, xlim[1]))
    ax.set_xlim(xlim[0]-0.1, xlim[1]+0.1)
    [s.set_visible(False) for s in [ax.spines['top'], ax.spines['left'], ax.spines['right']]]
    if print_text: 
        p_text = r'$p_{smash}$=%0.3f' % p_non_param  if p_non_param>=0.001 else r'$p_{smash}$<0.001'
        ax.text(0, ax.get_ylim()[1]+0.25, 'r={0:.2f}, {1:s}'.format(r_param,p_text), ha='center',va='top', color='k')
    plt.gca().grid(False,axis='x')
    plt.gca().set(xlabel='Correlation',ylabel='Density')

def plot_surf(met,met_fn,ax='',cmap='magma',vlow=0,vhigh=100,colorbar=False,fig_title=''):
    if colorbar:
        w, h = constants.LANDSCAPE_SIZE
        aspect = w / h
        fig = plt.figure(figsize=(5, 5/aspect))
        ax = fig.add_axes([0.075, 0, 0.85, 0.85])
        cax = fig.add_axes([0.44, 0.02, 0.12, 0.07])
    
    pscalar(
        file_out=met_fn,
        pscalars=met,
        orientation='landscape',
        hemisphere='left',
        vrange=(np.nanpercentile(met, vlow), np.nanpercentile(met, vhigh)),
        cmap=cmap
    )
    img = mpimg.imread(met_fn+'.png')
    cur_ax = ax if ax else plt
    cur_ax.imshow(img)
    cur_ax.axis('off')
    
    if colorbar:
        cnorm = clrs.Normalize(vmin=np.nanpercentile(met, vlow), vmax=np.nanpercentile(met, vhigh))  # only important for tick placing
        cmap = plt.get_cmap(cmap)
        cbar = colorbar.ColorbarBase(cax, cmap=cmap, norm=cnorm, orientation='horizontal')
        cbar.set_ticks([np.nanpercentile(met, vlow), np.nanpercentile(met, vhigh)])  # don't need to do this since we're going to hide them
        cax.get_xaxis().set_tick_params(length=0, pad=-2)
        cbar.set_ticklabels([])
        cax.text(-0.025, 0.4,str(np.nanmin(met).round(1)), ha='right', va='center', transform=cax.transAxes,#np.nanpercentile(met, vlow)
                 fontsize=10)
        cax.text(1.025, 0.4, str(np.nanmax(met).round(1)), ha='left', va='center', transform=cax.transAxes,#np.nanpercentile(met, vhigh)
                 fontsize=10)
        cbar.outline.set_visible(False)
    cur_ax.set_title(fig_title) if ax else cur_ax.title(fig_title)

def read_mask_nii(mask_fn,roi_labels=[],z_score=False,**nii_fns):
    """
    YOU, HELP ME!
    """
    masker = input_data.NiftiMasker(mask_img=mask_fn)
    data_dict = {}
    for prop,nii_fn in nii_fns.items():
#        print(f'{prop}:{nii_fn}')
        data_masked = masker.fit_transform(nii_fn)
        if((z_score) & (prop not in roi_labels)):
            data_masked_z = np.array([])
            for dcv in data_masked: 
                data_masked_z = stats.zscore(dcv)[np.newaxis,:] if data_masked_z.size==0 else np.concatenate((data_masked_z,stats.zscore(dcv)[np.newaxis,:]),axis=0)
            data_masked_z = np.nanmedian(data_masked_z,axis=0)
            data_dict[prop+'_z'] = data_masked_z
        if prop not in roi_labels:
            data_masked = np.nanmedian(data_masked,axis=0)     
        else:
            data_masked = data_masked.astype(int).flatten()
        data_dict[prop] = data_masked
    return data_dict
        
def multiple_joinplot(df,x,y,filtered_index_lists,np_null_dists,filter_labels,palette_regplot,color_scatterplot,xlabel='',ylabel='',xlim=None,ylim=None,s=0.1,legend_bbox_to_anchor=(-0.2,-0.5),plot_legend=True,mad_thr=None,prefix_legend_title=''):
    ps_legend_flag = True
    for fidx,filtered_index_list in enumerate(filtered_index_lists):
        df_filtered = df[filtered_index_list].copy()
        if mad_thr:
            df_filtered = df_filtered[remove_outliers(df_filtered[x].to_numpy(),mad_thr)]        
        rp,pp = stats.pearsonr(df_filtered.loc[df_filtered[x].notnull(),x],df_filtered.loc[df_filtered[x].notnull(),y])
        p_label = 'p_{smash}' if len(np_null_dists)>0 else 'p'
        pnp = nonparp(rp, np_null_dists[fidx]) if len(np_null_dists)>0 else pp
        pnp = pnp if pnp>0 else 0.001
        ps_legend_flag = (ps_legend_flag) & (pnp==0.001)
        label = f'{filter_labels[fidx]}={rp:.2f}({pnp:.3f})' if((pnp>0) & (~ps_legend_flag)) else f'{filter_labels[fidx]}={rp:.2f}'
        color_scatterplot_mod = color_scatterplot if len(color_scatterplot)>0 else palette_regplot[fidx]
        if fidx==0:
            g = sns.JointGrid(data=df_filtered, x=x, y=y)
            if xlim:
                g.ax_joint.set(xlim=xlim)
            if ylim:
                g.ax_joint.set(ylim=ylim)
            g.plot_joint(sns.regplot,line_kws={'linewidth':1,'color':palette_regplot[fidx]},scatter_kws={'s':s,'color':list(color_scatterplot_mod)},truncate=False)
            g.plot_marginals(sns.kdeplot,shade=False,linewidth=2,color=palette_regplot[fidx],label=label)
        else:
            if len(color_scatterplot)>0:
                sns.regplot(df_filtered[x],df_filtered[y],color=palette_regplot[fidx],ax=g.ax_joint,scatter=False,line_kws={'linewidth':1},truncate=False)
            else:
                sns.regplot(df_filtered[x],df_filtered[y],color=palette_regplot[fidx],ax=g.ax_joint,scatter=True,line_kws={'linewidth':1},truncate=False,scatter_kws={'s':s,'color':list(color_scatterplot_mod)})
            sns.kdeplot(x=df_filtered[x],shade=False,linewidth=2, ax=g.ax_marg_x,color=palette_regplot[fidx],label=label)
            sns.kdeplot(y=df_filtered[y],shade=False,linewidth=2, ax=g.ax_marg_y,color=palette_regplot[fidx])
    g.ax_joint.set(xlabel=xlabel,ylabel=ylabel)
    if plot_legend:
        legend_title = r'$Pearson\ r\ ('+p_label+'$<0.001)' if(ps_legend_flag) else r'$Pearson\ r\ ('+p_label+'$)'
        #legend_title = legend_title if len(prefix_legend_title)>0 else prefix_legend_title
        g.ax_marg_x.legend(loc='upper left',bbox_to_anchor=(1, 1.25))
        legend_handles = g.ax_marg_x.get_legend_handles_labels()
        g.ax_marg_x.get_legend().remove()
        leg = g.ax_joint.legend(legend_handles[0],legend_handles[1],bbox_to_anchor=legend_bbox_to_anchor, loc="lower left",title=legend_title,title_fontsize=16)#,title='orig_leg[1][7]'
        leg._legend_box.align = "left"