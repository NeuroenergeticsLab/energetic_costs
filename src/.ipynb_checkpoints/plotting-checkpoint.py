import matplotlib.colors as clrs
from matplotlib import colorbar
import matplotlib.image as mpimg
from wbplot import constants

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