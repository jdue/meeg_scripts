#import mne
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from ..utils import compute_misc
  
def plot_ica_artifacts(ica, inds, scores, which):
    """Barplot of ICA artifact scores.
    
    """
    keys = list(filter(lambda x: x.startswith(which+"/"), ica.labels_))
    
    fig, axes = plt.subplots(len(keys), 1)
    try:
        iter(axes)
    except TypeError:
        axes = [axes]
    if isinstance(scores, np.ndarray):
        scores = [scores]
        
    for i,ax in enumerate(axes):
        #key = [l for l in ica.labels_ if l.startswith("{}/{}".format(which.lower(),i))][0]
        rng = np.arange(ica.n_components_)
        ax.bar(rng, scores[i], color="b")
        idx = ica.labels_[keys[i]]
        if len(idx) > 0:
            ax.bar(rng[idx], scores[i][idx], color="r")
            ax.legend(["Non-"+which.upper(), which.upper()])
        else:
            ax.legend(["Non-"+which.upper()])
        ax.set_title("{} - Component(s) {}".format(keys[i], idx))
        ax.set_ylabel("Correlation Score")
        if i is len(axes)-1:
            ax.set_xlabel("ICA Components")
    fig.tight_layout()    
    return fig

def viz_bad_ica_components(ica, ch_type, eog_inds, ecg_inds, eog_scores, ecg_scores, eyeblinks_epoch, heartbeats_epoch):
    """Visualize the ICs deemed to be associated with EOG and ECG artifacts.
    
    """
    #### VISUALIZATION ####
    # Correlation scores
    fig = plot_ica_artifacts(ica, eog_inds, eog_scores, "eog")
    fig.savefig(ch_type+"_ICA_EOG_components.png", bbox_inches="tight")
    #fig.show()
    
    # not correlation score for ECG???
    fig = plot_ica_artifacts(ica, ecg_inds, ecg_scores, "ecg")
    fig.savefig(ch_type+"_ICA_ECG_components.png", bbox_inches="tight")
    #fig.show()
    
    # Plot how components relate to eyeblinks epochs
    figs = ica.plot_properties(eyeblinks_epoch, picks=eog_inds,
                               psd_args={"fmax": 30}, show=False)
    
    try:
        figs = iter(figs)
    except TypeError:
        figs = iter([figs])    
    for i,fig in enumerate(figs):
        fig.savefig("{}_ICA_EOG_{}".format(ch_type, i))
    #plt.show()
    
    figs = ica.plot_properties(heartbeats_epoch, picks=ecg_inds,
                               psd_args={"fmax": 80}, show=False)
    try: 
        figs = iter(figs)
    except TypeError:
        figs = iter([figs])    
    for i,fig in enumerate(figs):
        fig.savefig("{}_ICA_ECG_{}".format(ch_type, i))

    plt.close("all")
    
def viz_compare(raw_original, raw, raw_all, picks):
    """Compare raw ICA-REG and raw ALL-REG to the original raw data.
    
    """
    #### COMPARISON OF ICA-REG and ALL-REG
    # Only channel correlation with itself (bottom half diagonal of correlation
    # matrix)
    nchan = len(picks)
    
    corica = np.diag(np.corrcoef(raw_original.get_data()[picks],
                                 raw.get_data()[picks])[nchan:,:nchan])
    corrls = np.diag(np.corrcoef(raw_original.get_data()[picks],
                                 raw_all.get_data()[picks])[nchan:,:nchan])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(np.arange(nchan)-0.2, corica, width=0.4)
    ax.bar(np.arange(nchan)+0.2, corrls, width=0.4)
    ax.legend(["ICA-REG", "All-REG"])
    ax.set_title("Correlation With Original Data")
    ax.set_xlabel("Channels")
    ax.set_ylabel("Correlation Coefficient")
    #fig.savefig("Correlation_{}.png".format(ch_type))
    plt.show()
    
    # Plot raw timeseries
    raw_original.plot(duration=50, order=picks, title="Original Raw")
    raw.plot(duration=50, order=picks, title="ICA-REG Raw")
    raw_all.plot(duration=50, order=picks, title="ALL-REG Raw")
    
    
def viz_ar_bads(ar, epochs, show=False):
    
    reject_log = ar.get_reject_log(epochs)
    # There will be nans in the reject corresponding to non-functional channels
    
    chs = compute_misc.pick_func_channels(epochs.info)
    #labels = reject_log.labels[:, ar.picks_]
    bad_frac = (np.nan_to_num(reject_log.labels) > 0).mean(0)
    
    #bad_frac = ar.bad_segments.mean(0)
    
    nchan = len(ar.picks_)
    
    # Get average rejection threshold across sensors
    #thr = np.mean(ar._local_reject.threshes_["meg" if ch_type in ("mag", "grad") else "eeg"])
    #thr = [-thr, thr]

    # Plot bad segments, i.e., which channels were deemed bad in which epochs
    
    fig_bad = reject_log.plot(orientation='horizontal', show=False)
    """
    fig_bad = plt.figure() #figsize=(15,8)
    ax = fig_bad.add_subplot(111)
    ax.imshow(labels.T, cmap="gray")
    for bidx in np.where(reject_log.bad_epochs)[0]:
        # (x,y), width, heigh, options
        ax.add_patch(patches.Rectangle((bidx-0.5,-0.5), 1, nchan, facecolor="r",
                                       alpha=0.5))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Channel")
    ax.legend(["Good","Bad segments interp.","Bad segments not interp."])
    ax.set_title("Good (black), bads interp. (grey), bads not interp. (grey), bad epochs (red)")
    fig_bad.tight_layout()
    """
    # Plot fraction of bad trials per channel
    fig_frac,axes = plt.subplots(1, len(chs), figsize=(10,5))
    fig_frac.suptitle("Fraction of bad epochs per channel")
    for ax,k in zip(axes,chs):
        v = chs[k]
        ax.scatter(range(len(v)), bad_frac[v])
        ax.set_ylim(top=1 + np.abs(ax.get_ylim()[0]))
        ax.set_xlabel("Channel")
        ax.set_ylabel("Fraction")
        ax.set_title(k)
    
    """
    fig_frac = plt.figure()
    ax = fig_frac.add_subplot(111)
    ax.scatter(range(nchan), bad_frac, c=bad_frac)
    ax.set_ylim(top=1 + np.abs(ax.get_ylim()[0]))
    ax.set_xlabel("Channel")
    ax.set_ylabel("Fraction")
    ax.set_title("Fraction of bad trials per channel")
    """
    if show:
        plt.show()
    
    return fig_bad, fig_frac

def viz_compare_epochs(epochs_before, epochs_after, bad_epochs=None, show=False):
     #locs3d = np.array([ch['loc'][:3] for ch in epochs_orig.info["chs"]])
    #types = np.array([mne.io.pick.channel_type(epochs_orig.info, idx) for idx in None])
    #colors = mne.viz.evoked._handle_spatial_colors(locs3d, epochs_orig.info, range(len(picks)), "eeg", True, ax)
    
    if bad_epochs is None:
        bad_epochs = []
        
    nepoch, nchan, ntime = epochs_before.shape
    
    
    print("Plotting trials")
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(20,13))
    c = 0
    for i in range(nepoch):
        xas = range(ntime*i, ntime*(i+1))
        
        ax = axes[0]
        ax.plot(xas, epochs_before.get_data()[i].T, linewidth=0.25)
        
        if i not in bad_epochs:
            ax = axes[1]
            ax.plot(xas, epochs_after.get_data()[i-c].T, linewidth=0.25)
        else:
            c += 1
    
    txlim = axes[0].get_xlim()
    tylim = axes[0].get_ylim()
    
    for ax, which in zip(axes, ("before", "after")):
        #ax.set_color_cycle(colors)
        ax.set_xlim(txlim)
        ax.set_ylim(tylim)
        ax.set_title("Trials ({})".format(which))
        #ax.hlines(thr, *txlim, colors="b")
        ax.vlines(np.arange(ntime, nepoch*ntime, ntime), *tylim, linewidths=0.25, alpha=0.5)
    fig.tight_layout()
    if show:
        plt.show()
    
    return fig
    #plt.figure()
    #plt.gca().set_prop_cycle(colors[:2])
    #plt.plot(np.random.random((10,2)));plt.show() 
