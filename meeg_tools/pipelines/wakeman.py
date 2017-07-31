"""
Imports
"""

from autoreject import (compute_thresholds, LocalAutoRejectCV)
from datetime import datetime
    
from functools import partial
from glob import glob
import matplotlib.pyplot as plt
import mne
from mne.io.constants import FIFF
import numpy as np
import os
import os.path as op
import shutil
from warnings import warn

import sys
sys.path.append("C:\\Users\\jdue\\Google Drive\\PYTHON\\packages\\meeg_tools")

from meeg_tools.io import io_stuff
from meeg_tools.utils import compute_stuff
from meeg_tools.viz import visualize_stuff

from sklearn.model_selection import KFold
from scipy.stats import sem


def process_raw(fname_raw, channels=None, filt=None,
                        physiological_noise_correction='ica-reg'):
    """
    
    * set bad channels
    * rename channels
    
    * detect (squid) jumps
    * detect eyeblinks
    * detect heartbeats
    
    * filter (hp/lp/notch/chpi)
    
    * Rereference EEG to average
    
    * Calculate noise covariance
    
    * physiological noise correction
    ** ICA and regression of eyeblinks and heartbeats
    
    
    fname_raw : 
    channels : 
    filt : 
    noise_cov : 
    stim_delay : 
    phc :
    interactive : bool
    
    outdir : if None, use directory of input file...
    
    """
    assert isinstance(fname_raw, str) and op.isfile(fname_raw)
    basename, outdir, figdir = get_output_names(fname_raw)
    
    print("Reading data...", end="\t")
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    print("Done")
    
    if channels is not None:
        print("Relabelling channels and types")
        if 'rename' in channels:
            raw.rename_channels(channels['rename'])
        if 'change_types' in channels:
            raw.set_channel_types(channels['change_types'])
    
    
    # Get indices for different channel types (EEG, MAG, GRAD)
    chs = compute_stuff.pick_func_channels(raw.info)
    
    # EOG and ECG
    eog = mne.pick_types(raw.info, meg=False, eog=True, ref_meg=False)
    if any(eog):
        heog = mne.pick_channels(raw.info["ch_names"], ["HEOG"])
        veog = mne.pick_channels(raw.info["ch_names"], ["VEOG"])
    ecg = mne.pick_types(raw.info, meg=False, ecg=True, ref_meg=False)
    
    # Set bad channels
    raw.info["bads"] += channels['bads']
    print("The following channels were marked as bad:")
    print(raw.info["bads"])
    
    
         
    
    # If continuous HPI/MaxFiltering
    # Check if the data has been maxfiltered
    # raises RuntimeError if it has
    try:
        mne.preprocessing.maxwell._check_info(raw.info)
    except RuntimeError:
        hpi_channel = "STI201"
        chpi = mne.find_events(raw, hpi_channel, min_duration=10, verbose=False)
        
        if chpi.any():
            print("Annotating pre-cHPI samples as bad")
            onset = [raw.first_samp/raw.info["sfreq"]]
            # add 0.25 which is the approximate duration of the cHPI SSS transition noise
            chpi_duration = [(chpi[0][0]-raw.first_samp)/raw.info["sfreq"] + 2]
            description = ["bad_cHPI_off"]
            raw.annotations = mne.Annotations(onset, chpi_duration, description,
                                              orig_time=raw.info["meas_date"])
            
    #### Squid jumps ####
    
    # Annotate bad segments inplace
    print("Detecting squid jumps / high frequency stuff...")
    try:
        na = len(raw.annotations)
    except TypeError:
        na = 0
        ##################### HANDLE THIS #####################
        raise ValueError("handle this -> if no annotations exist")
           
    compute_stuff.detect_squid_jumps(raw)
    print("Annotated {} segments as bad".format(len(raw.annotations)-na))

    if any(eog):
        print("Detecting Eyeblinks")
        # Uses :
        #   [x.max()-x.min()] / 4 as threshold
        
        eyeblinks = mne.preprocessing.find_eog_events(raw, ch_name="VEOG")
        eyeblinks_epoch = mne.preprocessing.create_eog_epochs(raw, "VEOG", baseline=(None,None))
        
        eog_evoked = eyeblinks_epoch.average()
        
        fig = eog_evoked.plot(spatial_colors=True, show=False)
        fig.savefig(op.join(figdir, "Evoked_eyeblink.png"))
    
    if any(ecg):
        print("Detecting Heartbeats")
        heartbeats = mne.preprocessing.find_ecg_events(raw, ch_name="ECG")
        heartbeats_epoch = mne.preprocessing.create_ecg_epochs(raw, "ECG", baseline=(None,None))
        
        ecg_evoked = heartbeats_epoch.average()
        
        fig = ecg_evoked.plot(spatial_colors=True, show=False)
        fig.savefig(op.join(figdir, "Evoked_heartbeat.png"))
    
    plt.close("all")
    
    
    #### Interactively annotate/mark data ####
    """
    if interactive:
        
        # Manual detection of miscellaneous artifacts
        # Plot raw measurements with eyeblink events
        
        print("Vizualizing data for manual detection of miscellaneous artifacts")
        raw.plot(events=eyeblinks, duration=50, block=True)
        
        # Manual detection of squid jumps
        print("Visualizing data for manual squid jump detection")
        raw_sqj = raw.copy().pick_types(meg=True).filter(l_freq=1, h_freq=7)
        raw_sqj.plot(events=eyeblinks, duration=50, block=True)
        
        raw.info["bads"] += raw_sqj.info["bads"]
        raw.info["bads"] = np.unique(raw.info["bads"]).tolist()
        
        #raw.annotations.append( raw_sqj.annotations )
    
        plt.close("all")
    """
    
    
    #### Filtering ####
    
    # Filter also EOG and ECG channels such that frequencies are not introduced
    # again by the RLS regression...
    
    # see http://martinos.org/mne/dev/auto_tutorials/plot_background_filtering.html
    
    print("Filtering")
    
    alpha_level = 0.5
    try:
        tmin = 0+chpi_duration[0] # exclude the initial data annotated bad due to cHPI
    except NameError: # data was not cHPI annotated
        tmin = 0
    tmax = np.inf
    
    chpi_freqs = [hpi["coil_freq"][0] for hpi in raw.info["hpi_meas"][0]["hpi_coils"]]
    
    psd_kwargs = dict(average=False, line_alpha=alpha_level, show=False, verbose=False)
    
    # PSDs before filtering
    fig = raw.plot_psd(tmin, tmax, **psd_kwargs)
    fig.savefig(op.join(figdir, "PSD_0_NoFilter_0-nyquist.png"))
    fig = raw.plot_psd(tmin, tmax, fmin=0, fmax=50, **psd_kwargs)
    fig.savefig(op.join(figdir, "PSD_0_NoFilter_0-50.png"))
    
    # cHPI notch filtering
    #print("Removing cHPI noise at {} Hz from EEG channels".format(chpi_freqs))
    raw.notch_filter(chpi_freqs, picks=np.concatenate((chs["eeg"], eog, ecg)))
    
    # Notch filtering
    if filt['fnotch'] is not None:
        print("Removing line noise at {} Hz".format(filt['fnotch']))
        raw.notch_filter(filt['fnotch'])
        raw.notch_filter(filt['fnotch'], picks=np.concatenate((eog, ecg)))
    
    # 20, 26, 60 ???
    
    fig = raw.plot_psd(tmin, tmax, **psd_kwargs)
    fig.savefig(op.join(figdir, "PSD_1_Notch_0-nyquist.png"))
    
    # Highpass filtering
    if filt['fmin'] is not None:
        print("Highpass filtering at {} Hz".format(filt['fmin']))
        raw.filter(l_freq=filt['fmin'], h_freq=None)
        raw.filter(l_freq=filt['fmin'], h_freq=None, picks=np.concatenate((eog, ecg)))
        
        fig = raw.plot_psd(tmin, tmax, fmin=0, fmax=50, **psd_kwargs)
        fig.savefig(op.join(figdir, "PSD_2_NotchHigh_0-50.png"))
    
    # Lowpass filtering
    if filt['fmax'] is not None:
        print("Lowpass filtering at {} Hz".format(filt['fmax']))
        raw.filter(l_freq=None, h_freq=filt['fmax'])
        raw.filter(l_freq=None, h_freq=filt['fmax'], picks=np.concatenate((eog, ecg)))
        
        fig = raw.plot_psd(tmin, tmax, **psd_kwargs)
        fig.savefig(op.join(figdir,"PSD_3_NotchHighLow_0-nyquist.png"))
    plt.close("all")
    
    basename = 'f'+basename
    
    print("Rereferencing EEG data to average reference")
    #raw = mne.add_reference_channels(raw, "REF")
    raw.set_eeg_reference()

    # Physiological noise correction    
    if (physiological_noise_correction is not None) and (not any(np.concatenate((eog, ecg)))):
        warn("No EOG or ECG channels detected. Skipping ICA and RLS...")
        physiological_noise_correction = None
    else:
        basename = 'p'+basename
        print("Correcting for physiological artifacts")
    
    if physiological_noise_correction == "ica-reg":
        print("Using ICA and RLS")
        for ch, picks in chs.items():
            if ch == 'eeg':
                picks = np.concatenate((chs["eeg"], eog, ecg))
            nchan = len(picks)
            
            print("Processing {} {} channels".format(nchan, ch.upper()))
            #tol = 1e-6
            rank = raw.estimate_rank(0, None, picks=picks)
            #print("Estimated rank is {}".format(rank))
            print("Using {} principal components for ICA decomposition".format(rank))
            #rank = mne.utils.estimate_rank(raw[picks][0])
            
            # Fit ICA
            method = "extended-infomax"
            ica = mne.preprocessing.ICA(n_components=int(rank), method=method,
                                        noise_cov=None)
            ica.fit(raw, picks=picks)
            
            ica_name = op.join(outdir, "{}_{}-ica.fif".format(basename, ch))
            ica.save(ica_name)
            
            # Identify "bad" components
            print("Identifying bad components...")
            bad_idx = []
            if any(eog):
                eog_inds, eog_scores = ica.find_bads_eog(eyeblinks_epoch)
                bad_idx += eog_inds
                print("Found the following EOG related components: {}".format(eog_inds))
                
                if any(eog_inds):
                    # Plot correlation scores
                    fig = visualize_stuff.plot_ica_artifacts(ica, eog_inds, eog_scores, "eog")
                    eog_name = op.join(figdir, "ICA_EOG_components_{}".format(ch))
                    fig.savefig(eog_name, bbox_inches="tight")
                    
                    # Plot how components relate to eyeblinks epochs
                    figs = ica.plot_properties(eyeblinks_epoch, picks=eog_inds,
                                               psd_args={"fmax": 30}, show=False)
            
                    try:
                        figs = iter(figs)
                    except TypeError:
                        figs = iter([figs])    
                    for i,fig in enumerate(figs):
                        fig.savefig(op.join(figdir,"ICA_EOG_component_{}_{}".format(ch, i)))
              
            if any(ecg):        
                ecg_inds, ecg_scores = ica.find_bads_ecg(heartbeats_epoch)
                bad_idx += ecg_inds
                print("Found the following ECG related components: {}".format(ecg_inds))
                
                if any(ecg_inds):
                    fig = visualize_stuff.plot_ica_artifacts(ica, ecg_inds, ecg_scores, "ecg")
                    ecg_name = op.join(figdir, "ICA_ECG_components_{}".format(ch))
                    fig.savefig(ecg_name, bbox_inches="tight")
        
                    figs = ica.plot_properties(heartbeats_epoch, picks=ecg_inds,
                                               psd_args={"fmax": 80}, show=False)
                    
                    try: 
                        figs = iter(figs)
                    except TypeError:
                        figs = iter([figs])    
                    for i,fig in enumerate(figs):
                        fig.savefig(op.join(figdir, "ICA_ECG_component_{}_{}".format(ch, i)))
    
            bad_idx = np.unique(bad_idx)
                
            # Clean bad ICA components using RLS and reconstruct data from ICA
            print("Constructing sources")
            sources = compute_stuff.ica_unmix(ica, raw.get_data()[picks])
            
            print("Regressing bad components")
            signal = sources[bad_idx]
            reference = raw.get_data()[np.concatenate((eog, ecg))]
            signal = compute_stuff.rls(signal, reference)
            sources[bad_idx] = signal
            
            print("Reconstructing data")
            raw._data[picks] = compute_stuff.ica_mix(ica, sources)
            
            plt.close("all")
    
    elif physiological_noise_correction == "ica":
        raise ValueError("'ica' not implemented yet")
        
    elif physiological_noise_correction == "reg":
        print("Using RLS")    
        reference = raw.get_data()[np.concatenate((eog, ecg))]
        raw._data[picks] = compute_stuff.rls(raw.get_data()[np.concatenate((ch["meg"], ch["eeg"]))], reference)    
    
    elif physiological_noise_correction is None:
        print("Not correcting for physiological artifacts")
    
    else:
        IOError("Unknown argument '{}' for 'physiological_noise_correction'.".format(physiological_noise_correction))
    
    raw_name = op.join(outdir, "{}.fif".format(basename))
    raw.save(raw_name)
    
    return raw_name

def process_emptyroom(emptyroom_dir, subject_dir, raw, filt):
    """
    
    Use the same filt dict used to process the
    functional raw data.
    
    
    
    """
    # Find the empty room measurements closest to the acquisition date and use
    # that
    info = mne.io.read_info(raw)
    raw_date = datetime.fromtimestamp(info['meas_date'][0])
    
    acq = list()
    emptyroom_meas = glob(op.join(emptyroom_dir, '*raw_st.fif'))
    for i,er in enumerate(emptyroom_meas):
        er = op.basename(er)
        y = int(er[:2])+2000
        m = int(er[2:4])
        d = int(er[4:6])
        er_date = datetime(y,m,d)
        acq.append(np.abs((raw_date-er_date).days))
    fname = emptyroom_meas[np.argmin(acq)]
    raw = mne.io.read_raw_fif(fname, preload=True)
    
    print("Filtering")
    filt['fnotch'] = [f for f in filt['fnotch'] if f<=raw.info['sfreq']/2]
    print("Removing line noise at {} Hz".format(filt['fnotch']))
    raw.notch_filter(filt['fnotch'])
    
    if filt['fmin'] is not None:
        print("Highpass filtering at {} Hz".format(filt['fmin']))
        raw.filter(l_freq=filt['fmin'], h_freq=None)
    if filt['fmax'] is not None:
        print("Lowpass filtering at {} Hz".format(filt['fmax']))
        raw.filter(l_freq=None, h_freq=filt['fmax'])
    
    chs = compute_stuff.pick_func_channels(raw.info)
    cov_names = list()
    for ch, picks in chs.items():
        print("Computing noise covariance matrix for {} channels".format(ch.upper()))
        noise_cov = mne.compute_raw_covariance(raw, method="shrunk")
        noise_cov_name = op.join(subject_dir, 'MEEG', 'empty_room_{}_noise-cov.fif'.format(ch))
        noise_cov.save(noise_cov_name)
        cov_names.append(noise_cov_name)
    
    return cov_names

def process_epochs(fname_raw, event_codes, stim_chan=None, stim_delay=0, tmin=-0.2,
                  tmax=0.5):
    """
    
    raw : MNE raw object
    
    
    * Find events
    * Epoch
    ** optionally calculate noise cov from prestim baseline
    ** optionally calculate signal cov from entire epoch
    
    
    """
    assert isinstance(fname_raw, str) and op.isfile(fname_raw)
    basename, outdir, figdir = get_output_names(fname_raw)
    
    raw = mne.io.read_raw_fif(fname_raw)
    
    # uV, fT/cm, fT
    #scale = dict(eeg=1e6, grad=1e13, mag=1e15) # the 'scalings' dict from evoked.plot()

    
    
    print("Finding events")
    events, event_id = compute_stuff.find_events(raw, event_codes, stim_chan)
          
    # Compensate for stimulus_delay
    tmin += stim_delay
    tmax += stim_delay
    
    # we want baseline correction to get accurate noise cov estimate
    # this may distort the ERP/ERF though !!
    baseline = (tmin, stim_delay) # use stim_delay since we roll back the evoked axis
    
    print('Stimulus delay is {:0.0f} ms'.format(stim_delay*1e3))
    print('Epoching from {:0.0f} ms to {:0.0f} ms'.format(tmin*1e3,tmax*1e3))
    print('Baseline correction using {:0.0f} ms to {:0.0f} ms'.format(baseline[0]*1e3, baseline[1]*1e3))
    #noise_cov = mne.compute_raw_covariance(empty_room, tmin=0, tmax=None)
    
    print("Epoching...")
    
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                                 baseline, reject=None)
    
    epo_name = "{}-epo.fif".format(basename)
    epo_name = op.join(outdir, epo_name)
    epochs.save(epo_name)
    
    #ylim = [0, 0]    
    evoked = list()
    for condition in event_id.keys():
        # Average epochs
        evo = epochs[condition].average()
        evo.shift_time(-stim_delay)
        evoked.append(evo)
        
        # Get overall extrema for plotting
        #idx = evo.time_as_index(xlim)
        #mn = evo.data[:,idx[0]:idx[1]].min()
        #mx = evo.data[:,idx[0]:idx[1]].max()
        #mn = evo.data.min()
        #mx = evo.data.max()
        #if mn < ylim[0]:
        #    ylim[0] = mn
        #if mx > ylim[1]:
        #    ylim[1] = mx

    # Save data
    evo_name = "{}-ave.fif".format(basename)
    evo_name = op.join(outdir, evo_name)
    mne.evoked.write_evokeds(evo_name, evoked)
    
    # Get indices for different channel types (EEG, MAG, GRAD)
    chs = compute_stuff.pick_func_channels(evoked[0].info)
    for ch,picks in chs.items():
        for evo in evoked:
            condition = evo.comment
            
            evo_name = "{}_{}_{}".format(basename, ch, condition)
            evo_name = op.join(figdir, evo_name)
            
            fig = evo.plot(picks=picks, spatial_colors=True, exclude='bads',
                           show=False)
            fig.axes[0].set_title("Evoked '{}' [{}]".format(condition, ch.upper()))
            fig.savefig(evo_name+".png")
        plt.close('all')
    """
    epochs = dict()
    for ch, picks in chs.items():
        print("Epoching {} {} channels".format(len(picks), ch.upper()))
        epochs[ch] = mne.Epochs(raw, events, event_id, tmin, tmax, 
                                     baseline, picks,
                                     reject=None, preload=True)
        
        epo_name = "{}_{}-epo.fif".format(basename, ch)
        epochs[ch].save(op.join(outdir, epo_name))
        
        #ylim = [0, 0]    
        evoked = list()
        for condition in event_id.keys():
            # Average epochs
            evo = epochs[ch][condition].average()
            evo.shift_time(-stim_delay)
            evoked.append(evo)
            
            # Get overall extrema for plotting
            #idx = evo.time_as_index(xlim)
            #mn = evo.data[:,idx[0]:idx[1]].min()
            #mx = evo.data[:,idx[0]:idx[1]].max()
            #mn = evo.data.min()
            #mx = evo.data.max()
            #if mn < ylim[0]:
            #    ylim[0] = mn
            #if mx > ylim[1]:
            #    ylim[1] = mx

        # Save data
        evo_name = "{}_{}-ave.fif".format(basename, ch)
        evo_name = op.join(outdir, evo_name)
        mne.evoked.write_evokeds(evo_name, evoked)
    
        for evo in evoked:
            condition = evo.comment
            
            evo_name = "{}_{}_{}".format(basename, ch, condition)
            evo_name = op.join(figdir, evo_name)
            
            fig = evo.plot(spatial_colors=True, exclude='bads', show=False)
            fig.axes[0].set_title("Evoked '{}' [{}]".format(condition, ch.upper()))
            fig.savefig(evo_name+".png")
        plt.close('all')
    """
    
    return epo_name

def concat_epochs(fname_epochs, basename=None, outdir=None):
    """
    Concatenate list of epochs from filenames.
    
    """
    assert isinstance(fname_epochs, list)
    
    if basename is None:
        # Find basename as the parts which all input filenames share
        idx = list()
        bases = [op.basename(f).split('-')[0].split('_') for f in fname_epochs]
        nblocks = len(bases[0])
        for i in range(nblocks):
            bi = [b[i] for b in bases]
            if all( [ bi[0] == bix for bix in bi ]):
                idx.append(True)
            else:
                idx.append(False)
        basename = '_'.join(bases[0][i] for i in range(len(bases[0])) if idx[i])   

    if outdir is None:
        # Find outdir as the closest common directory of all inputs
        checkdirs = fname_epochs.copy()
        while outdir is None:
            pdir = [op.dirname(f) for f in checkdirs]
            if all( [pdir[0] == d for d in pdir ] ):
                outdir = pdir[0]
                break
            else:
                checkdirs = pdir
    epochs_name = op.join(outdir, basename+'-epo.fif')
    
    print('Loading epochs...', end=' ')
    epochs = [mne.read_epochs(f) for f in fname_epochs]
    print('Done')
    
    print('Concatenating epochs...', end=' ')
    epochs = mne.concatenate_epochs([epo for epo in epochs])
    print('Done')
    
    print('Saving concatenated epochs...', end=' ')
    epochs.save(epochs_name)
    print('Done')
    
    return epochs_name

def do_autoreject(fname_epochs, stim_delay=0):
    """
    
    epochs : dict
    
    obj_dir : str
        Directory in which to save the epochs
        
    
    """
    
    # We need to optimize two parameters by cross validation
    #   rho   : the maxmimum number of "bad" channels to interpolate)
    #   kappa : fraction of channels which has to be deemed bad for an epoch to be
    #           dropped
    assert isinstance(fname_epochs, str) and op.isfile(fname_epochs)
    basename, outdir, figdir = get_output_names(fname_epochs)
    basename = 'a'+basename
    
    epoch = mne.read_epochs(fname_epochs)
    
    chs = compute_stuff.pick_func_channels(epoch.info)
    
    print("Running AutoReject")
    print("")
    
    kappa = np.linspace(0, 1.0, 11)
    
    thresh_func = partial(compute_thresholds, method="random_search")
    
    ntrials = len(epoch)
    
    epochs = list()
    for ch,picks in chs.items():
        nchan = len(picks)
        
        epo = epoch.copy().pick_channels([epoch.ch_names[i] for i in picks])
        
        rho = np.round(np.arange(nchan/10, nchan/3, nchan/10)).astype(np.int)
        
        print("Channel type : {}".format(ch.upper()))
        print("Channels     : {}".format(nchan))
        print("Trials       : {}".format(ntrials))
        print("kappa        : {}".format(kappa))
        print("rho          : {}".format(rho))
        
        # Fit (local) AutoReject using CV
        ar = LocalAutoRejectCV(rho, kappa, thresh_func=thresh_func)
        cepo = ar.fit_transform(epo)
        
        # Save
        epo_name = "{}_{}-epo.fif".format(basename, ch)
        epo_name = op.join(outdir, epo_name)
        cepo.save(epo_name)
        epochs.append(epo_name)
        
        # Evoked response after AR
        evoked = list()
        for condition in cepo.event_id.keys():
            evo = cepo[condition].average()
            evo.shift_time(-stim_delay)
            evoked.append(evo)

        evo_name = "{}_{}-ave.fif".format(basename, ch)
        evo_name = op.join(outdir, evo_name)
        mne.evoked.write_evokeds(evo_name, evoked)
        
        # Visualize results of autoreject
        
        # Bad segments
        fig_bad, fig_frac = visualize_stuff.viz_ar_bads(ar)
        fig_name = op.join(figdir, "Epochs_bad_segments_{}.png".format(ch))
        fig_bad.savefig(fig_name, bbox_inches="tight")
        fig_name = op.join(figdir, "Epochs_bad_fractions_{}.png".format(ch))
        fig_frac.savefig(fig_name, bbox_inches="tight")
        
        # Check for bad channels
        bad_cutoff = 0.5
        bad_frac = ar.bad_segments.mean(0)
        possible_bads = [cepo.ch_names[bad] for bad in np.where(bad_frac>bad_cutoff)[0]]
        
        for evo in evoked:                
            condition = evo.comment
            
            evo_name = "{}_{}_{}".format(basename, ch, condition)
            evo_name = op.join(figdir, evo_name)
            
            fig = evo.plot(spatial_colors=True, exclude='bads', show=False)
            fig.axes[0].set_title("Evoked '{}' [{}]".format(condition, ch.upper()))
            fig.savefig(evo_name+".png")
        
            if any(possible_bads):
                # Evoked response before AR
                evoo = epo[condition].average()
                evoo.shift_time(-stim_delay)
                evoo.plot(spatial_colors=True, exclude='bads', show=False)
                
                # Plot the bad channels before and after AR
                tb = evoo.copy()
                ta = evo.copy()
                tb.info["bads"] = possible_bads
                ta.info["bads"] = possible_bads
                
                fig = tb.plot(spatial_colors=False, exclude=[], show=False)
                fig.axes[0].set_title("Evoked '{}' [{}] before AutoReject".format(condition, ch.upper()))
                fig.savefig(evo_name+"-bads-before.png")
                fig = ta.plot(spatial_colors=False, exclude=[], show=False)
                fig.axes[0].set_title("Evoked '{}' [{}] after AutoReject".format(condition, ch.upper()))
                fig.savefig(evo_name+"-bads-after.png")
    
    return epochs

    """
    for ch, epoch in epochs.items():
        nchan = len(epoch.ch_names)
        
        
        rho = np.round(np.arange(nchan/10, nchan/3, nchan/10)).astype(np.int)
        
        print("Channel type : {}".format(ch.upper()))
        print("Channels     : {}".format(nchan))
        print("Trials       : {}".format(ntrials))
        print("kappa        : {}".format(kappa))
        print("rho          : {}".format(rho))
        
        # Fit (local) AutoReject using CV
        ar[ch] = LocalAutoRejectCV(rho, kappa, thresh_func=thresh_func)
        epochs[ch] = ar[ch].fit_transform(epoch)
        
        # Save     
        if obj_dir is not None:
            epo_name = "{}_{}-epo.fif".format(fname_base, ch)
            epochs[ch].save(op.join(obj_dir, epo_name))
        
        if obj_dir is not None and fig_dir is not None:
            fig_bad, fig_frac = visualize_stuff.viz_ar_bads(ar[ch])
            
            fig_name = op.join(fig_dir, "Epochs_bad_segments_{}.png".format(ch))
            fig_bad.savefig(fig_name, bbox_inches="tight")
            fig_name = op.join(fig_dir, "Epochs_bad_fractions_{}.png".format(ch))
            fig_frac.savefig(fig_name, bbox_inches="tight")
        
            # Average : Save evoked and figures
            evoked = list()
            for condition in epochs[ch].event_id.keys():
                # Average epochs
                evo = epochs[ch][condition].average()
                evo.shift_time(-stim_delay)
                evoked.append(evo)

            # Save data
            evo_name = "{}_{}-ave.fif".format(fname_base, ch)
            evo_name = op.join(obj_dir, evo_name)
            mne.evoked.write_evokeds(evo_name, evoked)

            bad_cutoff = 0.5
            bad_frac = ar[ch].bad_segments.mean(0)
            possible_bads = [evo.info["ch_names"][bad] for bad in np.where(bad_frac>bad_cutoff)[0]]
            
            for evo in evoked:                
                condition = evo.comment
                
                evo_name = "{}_{}_{}".format(fname_base, ch, condition)
                evo_name = op.join(fig_dir, evo_name)
                
                fig = evo.plot(spatial_colors=True, exclude='bads', show=False)
                fig.axes[0].set_title("Evoked '{}' [{}]".format(condition, ch.upper()))
                fig.savefig(evo_name+".png")
            
                if any(possible_bads):
                    # Evoked response without artifact correction
                    evoked_orig = epochs_orig[ch][condition].average()
                    evoked_orig.shift_time(-stim_delay)
                    evoked_orig.plot(spatial_colors=True, exclude='bads', show=False)
                    
                    tb = evoked_orig.copy()
                    ta = evo.copy()
                    tb.info["bads"] = possible_bads
                    ta.info["bads"] = possible_bads
                    
                    fig = tb.plot(spatial_colors=False, exclude=[], show=False)
                    fig.axes[0].set_title("Evoked '{}' [{}] before AutoReject".format(condition, ch.upper()))
                    fig.savefig(evo_name+"-bads-before.png")
                    fig = ta.plot(spatial_colors=False, exclude=[], show=False)
                    fig.axes[0].set_title("Evoked '{}' [{}] after AutoReject".format(condition, ch.upper()))
                    fig.savefig(evo_name+"-bads-after.png")
            
            plt.close("all")   
            print("Done")
            print("")
    
    return epochs
    """

def cov_epochs(epochs, cov_type='noise'):
    """
    Noise or signal covariance from epochs.
    """

    assert isinstance(epochs, str) and op.isfile(epochs)
    basename, outdir, figdir = get_output_names(epochs)
    
    epochs = mne.read_epochs(epochs)

    # Covariance window
    if cov_type == 'noise':
        win = epochs.baseline
    elif cov_type == 'signal':
        win = (None, None)
    else:
        raise TypeError("cov_type must be 'noise' or 'signal'")

    cov = mne.compute_covariance(epochs, tmin=win[0], tmax=win[1],
                                 method="shrunk")
    
    cov_name = op.join(outdir, "{}_{}-cov.fif".format(basename, cov_type))
    cov.save(cov_name)
        
    return cov_name
    """
    cov = dict()
    for ch in epochs.keys():
        basename, outdir, figdir = get_output_names(epochs[ch])
        epo = mne.read_epochs(epochs[ch])
        
        # Covariance window
        if cov_type == 'noise':
            win = epochs.baseline
        elif cov_type == 'signal':
            win = (None, None)
        else:
            raise TypeError("cov_type must be 'noise' or 'signal'")

        cov[ch] = mne.compute_covariance(epo, tmin=win[0], tmax=win[1],
                                         method="shrunk")
        
        op.join(obj_dir, "{}_{}_{}-cov.fif".format(basename, cov_type))
        cov[ch].save()
    
    return cov
    """
    
def do_xdawn(epochs, signal_cov, stim_delay=0):
    """
    
    """
    
    assert isinstance(epochs, str) and op.isfile(epochs)
    basename, outdir, figdir = get_output_names(epochs)
    basename = 'x'+basename
    
    epochs = mne.read_epochs(epochs)
    signal_cov = mne.read_cov(signal_cov)
    
    #scale = dict(eeg=1e6, grad=1e13, mag=1e15)
    
    print('Denoising using xDAWN')
    n_components, fig = xdawn_cv(epochs, signal_cov)
    fig.set_size_inches(10,10)
    fig.savefig(op.join(figdir, '{}_xdawn_cv.png').format(basename))
    
    data = np.zeros_like(epochs.get_data())
    for condition, eidv in epochs.event_id.items():
        # Fit each condition separately
        print(condition)
        xd = mne.preprocessing.Xdawn(n_components[condition], signal_cov,
                                     correct_overlap=False)
        xd.fit(epochs[condition])
        x = xd.apply(epochs[condition])[condition].get_data()
        data[epochs.events[:,2]==eidv] = x
        
        # Component time series
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot((epochs.times-stim_delay)*1e3,
                xd.transform(epochs[condition].get_data()).mean(0).T)#*scale[ch])
        ax.set_xlabel('Time (ms)')
        plt.legend([str(i) for i in range(xd.n_components)])
        ax.set_title("xDAWN time courses '{}'".format(condition))
        fig.savefig(op.join(figdir, '{}_xdawn_timecourses_{}.png'.format(basename, condition)))
    
        # Component patterns
        nrows = 2
        ncols = 4
        fig, axes = plt.subplots(2, 4)
        i = 0
        for row in range(nrows):
            for col in range(ncols):
                mne.viz.plot_topomap(xd.patterns_[condition].T[i], epochs.info,
                                     axes=axes[row,col], show=False)
                i += 1
        fig.suptitle("xDAWN patterns '{}'".format(condition))
        fig.tight_layout()
        fig.savefig(op.join(figdir, '{}_xdawn_patterns_{}.png'.format(basename, condition)))            
    
    # Update epochs
    epochs._data = data.copy()
    
    # Save epochs
    epo_name = "{}-epo.fif".format(basename)
    epo_name = op.join(outdir, epo_name)
    epochs.save(epo_name)

    # Evoked    
    evoked = list()
    for condition in epochs.event_id.keys():
        evo = epochs[condition].average()
        evo.shift_time(-stim_delay)
        evoked.append(evo)
            
    # Save data
    evo_name = "{}-ave.fif".format(basename)
    evo_name = op.join(outdir, evo_name)
    mne.evoked.write_evokeds(evo_name, evoked)
    
    for evo in evoked:                
        condition = evo.comment
        
        evo_name = "{}_{}".format(basename, condition)
        evo_name = op.join(figdir, evo_name)
        
        fig = evo.plot(spatial_colors=True, exclude='bads', show=False)
        fig.axes[0].set_title("Evoked '{}'".format(condition))
        fig.savefig(evo_name+".png")
            
    plt.close('all')
        
    return epo_name
    
    """
    #n_components = 8
    for ch in epochs.keys():
        print('Channels : {}'.format(ch.upper()))
        epo = epochs[ch]
        
        n_components, fig = xdawn_cv(epo, signal_cov[ch])
        fig.savefig(op.join(fig_dir, 'xdawn_cv_{}.png'.format(ch)))
        
        # Fit each condition separately
        data = np.zeros_like(epo.get_data())
        for condition, eidv in epo.event_id.items():
            print(condition)
            xd = mne.preprocessing.Xdawn(n_components[condition], signal_cov[ch], correct_overlap=False)
            xd.fit(epo[condition])
            x = xd.apply(epo[condition])[condition].get_data()
            data[epo.events[:,2]==eidv] = x
            
            if fig_dir is not None:
                # Component time series
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot((epo.times-stim_delay)*1e3,
                        xd.transform(epo[condition].get_data()).mean(0).T*scale[ch])
                ax.set_xlabel('Time (ms)')
                plt.legend([str(i) for i in range(xd.n_components)])
                ax.set_title("xDAWN time courses '{}' [{}]".format(condition, ch.upper()))
                fig.savefig(op.join(fig_dir, 'xdawn_timecourses_{}_{}.png'.format(ch, condition)))
                                
                # Component patterns
                nrows = 2
                ncols = 4
                fig, axes = plt.subplots(2, 4)
                i = 0
                for row in range(nrows):
                    for col in range(ncols):
                        mne.viz.plot_topomap(xd.patterns_[condition].T[i], epo.info, axes=axes[row,col], show=False)
                        i += 1
                fig.suptitle("xDAWN patterns '{}' [{}]".format(condition, ch.upper()))
                fig.tight_layout()
                fig.savefig(op.join(fig_dir, 'xdawn_patterns_{}_{}.png'.format(ch, condition)))            
        
        # Update epochs
        epochs[ch]._data = data.copy()
        
        # Save epochs
        if obj_dir is not None:
            epo_name = "{}_{}-epo.fif".format(fname_base, ch)
            epochs[ch].save(op.join(obj_dir, epo_name))
        
        if obj_dir is not None and fig_dir is not None:
            epo = epochs[ch] 
            evoked = list()
            for condition in epo.event_id.keys():
                # Average epochs
                evo = epo[condition].average()
                evo.shift_time(-stim_delay)
                evoked.append(evo)
                    
            # Save data
            evo_name = "{}_{}-ave.fif".format(fname_base, ch)
            evo_name = op.join(obj_dir, evo_name)
            mne.evoked.write_evokeds(evo_name, evoked)
            
            for evo in evoked:                
                condition = evo.comment
                
                evo_name = "{}_{}_{}".format(fname_base, ch, condition)
                evo_name = op.join(fig_dir, evo_name)
                
                fig = evo.plot(spatial_colors=True, exclude='bads', show=False)
                fig.axes[0].set_title("Evoked '{}' [{}]".format(condition, ch.upper()))
                fig.savefig(evo_name+".png")
                
        plt.close('all')
    
    return epochs
    """
def xdawn_cv(epochs, signalcov, component_grid=None, nfolds=5, cv_curve=True):
    """Use cross validation to find the optimal number of xDAWN components to
    project the epochs onto.
    
    epochs : MNE.Epochs object
    
    signalcov : 
        
    component_grid : 
    
    folds : 
    
    """
    if component_grid is None:
        component_grid = range(1,11)
        
    kf = KFold(n_splits=nfolds, shuffle=True)
    
    print('{} fold cross validation'.format(nfolds))
    
    n_components_cv = dict().fromkeys(epochs.event_id.keys())
    RMSE = dict().fromkeys(epochs.event_id.keys())

    if cv_curve:
        fig, axes = plt.subplots(nrows=1, ncols=len(epochs.event_id), sharey=True)
        try:
            len(axes)
        except TypeError:
            axes = [axes]
    for eid,ax in zip(epochs.event_id, axes):
        print('Event', eid)
        epo = epochs[eid]
        RMSE[eid] = np.zeros((nfolds,len(component_grid)))
        
        for i, (train, test) in enumerate(kf.split(range(len(epo)))):
            for j, n_components in enumerate(component_grid):
                train_set = epo.copy().drop(test)
                test_set = epo.copy().drop(train)
                
                xd = mne.preprocessing.Xdawn(n_components, signalcov, correct_overlap=False)
                xd.fit(train_set)
                trainx = xd.apply(train_set)[eid]

                RMSE[eid][i,j] = np.sqrt(((trainx.get_data().mean(0)-test_set.get_data().mean(0))**2).mean())
        
        # Mean across folds
        mean = RMSE[eid].mean(0)
        SE = sem(RMSE[eid],0) # RMSE.std(0,ddof=1) / np.sqrt(5)
        
        # Find parameter using the 'one standard error rule' for n_components_cving the
        # most parsimonious model, i.e., the simplest model within one SE of
        # the 'best' model
        idx_opt = mean.argmin()
        idx_sel = np.where(mean < mean[idx_opt]+SE[idx_opt])[0][0]
        
        # The number of components
        n_components_cv[eid] = component_grid[idx_sel]
        
        if cv_curve:
            ax.plot(component_grid, mean, color='black')
            ax.fill_between(component_grid, mean-SE, mean+SE, facecolor='gray',
                             alpha=0.5)
            ax.plot([component_grid[0], component_grid[-1]],
                    [mean[idx_opt], mean[idx_opt]], color='r', linestyle='--')
            ax.plot([component_grid[0], component_grid[-1]],
                    [mean[idx_opt]+SE[idx_opt], mean[idx_opt]+SE[idx_opt]],
                    color='b', linestyle='--')
            ax.legend(['RMSE', 'RMSE (optimal)', 'RMSE (selected)'])
            ax.set_xlabel('N components')
            ax.set_ylabel('RMSE')
            ax.set_title(eid)
    
    if cv_curve:
        return n_components_cv, fig
    else:
        return n_components_cv
    


def setup_runs(subject_dir):
    """
    
    """
    meg_dir = op.join(subject_dir, 'MEEG')
    
    # Move runs to separate folders
    dsts = list()
    for r in glob(op.join(meg_dir, 'run_*_sss.fif')):
        basename = op.basename(r)
        basesplit, _ = op.splitext(basename)
        
        rundir = op.join(meg_dir, basesplit)    # base directory for this run
        figdir = op.join(rundir, 'figures') # keep figures here
        
        if not op.exists(rundir):
            os.makedirs(rundir, exist_ok=True)
        if not op.exists(figdir):
            os.makedirs(figdir, exist_ok=True)
        
        dst = op.join(rundir, basename)
        shutil.move(r, dst)
        dsts.append(dst)
    
    # If they have already been moved, fetch them
    if not any(dsts):
        for r in glob(op.join(meg_dir, 'run_*_sss')):
            assert op.isdir(r)
            r = glob(op.join(r, 'run_*_sss.fif'))[0]
            dsts.append(r)
            
    return dsts

def get_output_names(fname_raw):
    
    outdir = op.dirname(fname_raw)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.mkdir(figdir)
    basename, _ = op.splitext(op.basename(fname_raw)) 
    basename = basename.split('-')[0]
    
    return basename, outdir, figdir

#%%    
"""  
def estimate_
        if estimate_noise_cov:
            print("Estimating noise covariance in {}".format(baseline))
            noise_cov[ch] = mne.compute_covariance(epochs[ch],tmin=baseline[0],
                                           tmax=baseline[1], method="shrunk")
            
            noise_cov[ch].save(op.join(savedir, "{}_{}_noise-cov.fif".format(raw_base, ch)))
        
            # Save figures
            figs = noise_cov[ch].plot(epochs[ch].info, show=False)
            for fig,which in zip(figs, ["mat", "eig"]):
                fig.set_size_inches(10,10)
                fig.savefig(op.join(evo_fig, "{}_{}_noise-cov_{}.png".format(raw_base, ch, which)))
            
        if estimate_signal_cov or xdawn:
        # Use entire epoch (i.e., all signal + noise) or 0 to end?
            print("Estimating signal covariance in {}".format((tmin, tmax)))
            signal_cov[ch] = mne.compute_covariance(epochs[ch],tmin=None,
                                           tmax=None, method="shrunk")
            
            signal_cov[ch].save(op.join(savedir, "{}_{}_signal-cov.fif".format(raw_base, ch)))
            
            # Save figures
            figs = signal_cov[ch].plot(epochs[ch].info, show=False)
            for fig,which in zip(figs, ["mat", "eig"]):
                fig.set_size_inches(10,10)
                fig.savefig(op.join(evo_fig, "{}_{}_signal-cov_{}.png".format(raw_base, ch, which)))
            
    plt.close("all")     
    
    print("Done")
""" 




def process_coregistration(fname_raw, fname_mri_fids, fname_scalp, subject_dir):
    """
    
    Write sensors, digitized points, fiducials (MRI and MEG) in head
    coordinates.
    
    fname_raw : 
        The raw file containing digitized points.
    fname_mri_fids : 
        Text file containing coordinates of the fiducials in MRI space.
    fname_scalp : 
        Filename of the scalp mesh. The points (e.g., EEG electrodes) are
        projected onto this surface.
    
    
    """
    basename, outdir, _ = get_output_names(fname_raw)
    
    meeg_dir = op.join(subject_dir, 'MEEG')
    if not op.exists(meeg_dir):
        os.makedirs(meeg_dir)
    coreg_dir = op.join(meeg_dir, 'coreg')
    if not op.exists(coreg_dir):
        os.makedirs(coreg_dir)
    
    info = mne.io.read_info(fname_raw)
    
    # Fiducials MRI
    fids_mri_coords = np.loadtxt(fname_mri_fids, usecols=(0,1,2))
    fids_mri_coords /= 1000 # convert to m
    fids_mri_labels = np.loadtxt(fname_mri_fids, dtype="S3", usecols=3)
    fids_mri = dict()
    for i,ii in zip(fids_mri_labels, fids_mri_coords):    
        fids_mri[str(i,"utf-8").lower()] = ii
    
    vertices, faces = io_stuff.read_surface(fname_scalp)
    vertices /= 1000 
    scalp = dict(v=vertices, f=faces)
    
    print('Coregistering MRI to head coordinates')
    # Get the transformation
    trans, eeg_locs_proj, digs_proj = compute_stuff.get_mri2head_transform(info, fids_mri, scalp,
                                                 rm_points_below_nas=True)
    
    trans_name = op.join(meeg_dir, 'mri_head-trans.fif'.format(basename))
    trans.save(trans_name)

    # Transform from MRI to head coordinates
    # Convert back to mm and write
    
    # MRI fiducials in head coordinates
    mri_fids_base = op.basename(fname_mri_fids)
    mri_fids_base, _ = op.splitext(mri_fids_base)
    fids_mri = np.asarray([v for v in fids_mri.values()])
    mri_fids_head = mne.transforms.apply_trans(trans, fids_mri)
    vtk = io_stuff.as_vtk(mri_fids_head*1e3)
    vtk.tofile(op.join(coreg_dir, mri_fids_base+'_headcoord'))
    
    # Scalp mesh in head coordinates
    scalp_base = op.basename(fname_scalp)
    scalp_base, _ = op.splitext(scalp_base)
    scalp['v'] = mne.transforms.apply_trans(trans, scalp['v'])
    vtk = io_stuff.as_vtk(scalp['v']*1e3, scalp['f'])
    vtk.tofile(op.join(coreg_dir, scalp_base+'_headcoord'))
    
    # Sensors
    locs = compute_stuff.get_sensor_locs(info)
    for ch, loc in locs.items():
        vtk = io_stuff.as_vtk(loc*1e3)
        vtk.tofile(op.join(coreg_dir, '{}_locs_headcoord'.format(ch)))
        
    # Digitized points
    digs, _, fids = compute_stuff.get_digitized_points(info)
    vtk = io_stuff.as_vtk(digs*1e3)
    vtk.tofile(op.join(coreg_dir, 'digs_headcoord'))
    vtk = io_stuff.as_vtk(np.asarray([v for v in fids.values()])*1e3)
    vtk.tofile(op.join(coreg_dir, 'meg_fids_headcoord'))

    # EEG electrodes projected onto scalp
    vtk = io_stuff.as_vtk(eeg_locs_proj*1e3)
    vtk.tofile(op.join(coreg_dir, 'eeg_locs_proj_headcoord'))

    # Digitized points projected onto scalp
    vtk = io_stuff.as_vtk(digs_proj*1e3)
    vtk.tofile(op.join(coreg_dir, 'digs_proj_headcoord'))
    
    return trans_name

    
"""
# Functional data
fname = "C:\\Users\\jdue\\Documents\\phd_data\\meeg_pipeline\\MRI_sub003\\sub003_run_01_mod_raw_sss.fif"

# Surfaces for BEM
# Perhaps not more than ~5000 vertices per surface
surf_dir = 'C:\\Users\\jdue\\Documents\\phd_data\\meeg_pipeline\\MRI_sub003'
fname_csf = op.join(surf_dir, 'csf.stl')
fname_bone = op.join(surf_dir, 'bone.stl')
fname_skin = op.join(surf_dir, 'skin.stl')

# Transformation between MRI and head coordinates
# trans should map between head and MRI coordinates (either way is okay)
# the data file (raw) already includes the device -> head mapping
fname_trans = op.join(surf_dir,'sub003_mri_head-trans.fif')

# Source space
fname_src = [op.join(surf_dir, 'lh.central.T1fs_conform.gii'),
             op.join(surf_dir, 'rh.central.T1fs_conform.gii')]
"""
   
def process_sourcespace(sourcespace_files, basename):
    
    print('Preparing source space...')
    src = io_stuff.sourcespace_from_files(sourcespace_files)
    src_name = basename+'-src.fif'
    src.save(src_name)
    return src
    
def process_bem(brain, skull, scalp, basename):
    """
    
    """
    
    # Setup some constants
    ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN,
           FIFF.FIFFV_BEM_SURF_ID_SKULL,
           FIFF.FIFFV_BEM_SURF_ID_HEAD]
    
    conductivities = (0.3, 0.006, 0.3)
    
    # Read surfaces for BEM
    surfs = []
    for surf in (brain, skull, scalp):
        v,f = io_stuff.read_surface(surf)
        surfs.append(dict(rr=v, tris=f, ntri=len(f), use_tris=f, np=len(v)))
    
    
    print('Preparing BEM model...')
    bem = mne.bem._surfaces_to_bem(surfs, ids, conductivities)
    print('Writing BEM model...')
    bem_name = basename+"-surfs.fif"
    mne.write_bem_surfaces(bem_name, bem)
    
    print('Computing BEM solution...')
    bem = mne.make_bem_solution(bem)
    print('Writing BEM solution...')
    bemsol_name = basename+"bem-sol.fif"
    mne.write_bem_solution(bemsol_name, bem)
    
    return bem
 
def process_forward(raw, src, bem, trans, basename):    
    """
    
    """
    if isinstance(raw, str):
        raw = io_stuff.read_data(raw)
    if isinstance(bem, str):
        bem = mne.read_bem_solution(bem)
    if isinstance(src, str):
        src = mne.read_source_spaces(src)
    if isinstance(trans, str):
        trans = mne.read_trans(trans)
    
    
    # MNE automatically projects EEG electrodes onto the scalp surface
    print('Computing forward solution')
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=True, mindist=0, n_jobs=2)
    
    # mne.convert_forward_solution(fwwd, surf_ori, force_fixed)
    print('Writing forward solution')
    fwd_name = basename+"-fwd.fif"
    mne.write_forward_solution(fwd_name, fwd)
    
    return fwd

def get_rawfile(subject_dir):
    data = glob(op.join(subject_dir, 'MEEG', 'run*sss.fif'))
    if len(data) > 0:
        return data[0]
    else:
        dirs = glob(op.join(subject_dir, 'MEEG', 'run*sss'))
        data = glob(op.join(dirs[0], 'run*sss.fif'))
        return data[0]

def get_mrifids(subject_dir):
    mrifids = op.join(subject_dir, 'T1', 'mri_fids.txt')
    return mrifids

def get_surface(subject_dir, name):
    name = glob(op.join(subject_dir, 'Surf', name))
    return name[0]

def get_in_meeg(subject_dir, pattern):
    if '*' in pattern:
        files = glob(op.join(subject_dir, 'MEEG', pattern))
    else:
        files = op.join(subject_dir, 'MEEG', pattern)
    return files
    

def fit_dipole(evoked, noise_cov, bem, trans, times):
    """
    
    
    
    """
    # I/O
    basename, outdir, figdir = get_output_names(evoked)
    evoked = mne.read_evokeds(evoked)
    
    if isinstance(noise_cov, str):
        noise_cov = mne.read_cov(noise_cov)
    if isinstance(bem, str):
        bem = mne.read_bem_solution(bem)
    if isinstance(trans, str):
        trans = mne.read_trans(trans)
    if isinstance(times, float):
        times = [times, times]
    assert isinstance(times, list) and len(times) is 2
    
    
    
    dips = list()
    for evo in evoked:
        print('Fitting dipole to condition {}'.format(evo.comment))
        #print('Fitting to time {} ms'.format([int(np.round(t*1e3)) for t in time]))
        evoc = evo.copy()
        evoc.crop(*times)
        dip, res = mne.fit_dipole(evoc, noise_cov, bem, trans)
        dips.append(dip)
        
        # Find best fit (goodness of fit)
        best_idx = np.argmax(dip.gof)
        best_time = dip.times[best_idx]
        best_time_ms = best_time*1e3
        
        # Crop evoked and dip
        evoc.crop(best_time, best_time)
        dip.crop(best_time, best_time)
        
        print('Found best fitting dipole (max GOF) at {:0.0f} ms'.format(best_time_ms))
        print('Estimating time course by fixing position and orientation')

        # Time course of the dipole with highest GOF  
        # dip_fixed.data[0] : dipole time course
        # dip_fixed.data[1] : dipole GOF (how much of the total variance is explained by this dipole)              
        dip_fixed = mne.fit_dipole(evo, noise_cov, bem, trans,
                           pos=dip.pos[best_idx], ori=dip.ori[best_idx])[0]
        fig = dip_fixed.plot(show=False)
        fig.suptitle('Dipole with max GOF at {:0.0f} ms'.format(best_time_ms))
        fig.set_size_inches(8,10)
        fig.show()
        
        # Plot residual field from dipole
        # Make forward solution for dipole
        fwd, stc = mne.make_forward_dipole(dip, bem, evo.info, trans)
        # Project to sensors
        pred_evo = mne.simulation.simulate_evoked(fwd, stc, evo.info, cov=noise_cov)
        # Calculate residual
        res_evo = mne.combine_evoked([evoc, -pred_evo], weights='equal')
        
        # Get min/max for plotting
        ch_type = mne.channels._get_ch_type(evo,None)
        scale = mne.defaults._handle_default('scalings')[ch_type]
        evo_cat = np.concatenate((evoc.data, pred_evo.data, res_evo.data))
        vmin = evo_cat.min()*scale
        vmax = evo_cat.max()*scale
        
        # Plot topomaps
        fig, axes = plt.subplots(1,4)
        plot_params = dict(vmin=vmin, vmax=vmax, times=best_time, colorbar=False)
        evoc.plot_topomap(time_format='Observed', axes=axes[0], **plot_params)
        pred_evo.plot_topomap(time_format='Predicted',axes=axes[1], **plot_params)
        plot_params['colorbar'] = True
        res_evo.plot_topomap(time_format='Residual',axes=axes[2], **plot_params)
        fig.suptitle('Residual field from dipole at {:.0f} ms'.format(best_time_ms))
        fig.set_size_inches(10,4)
        fig.show() # savefig()
        
        # Plot on MRI
        
        # Dipole positions and vectors are given in head coordinates
        dip_mri = mne.transforms.apply(np.linalg.inv(trans['trans'], ))
        
        dip.plot_amplitudes()
        dip.plot()
        
        # check out code in 
        # mne/viz/_3d.py - _plot_dipole_mri_orthoview
    
    return 

def do_inverse(evoked, noise_cov, fwd, times, method='dSPM', crop=True):
    """
    
    time : float | list
        Time (time window) at which the SNR is estimated.
    
    """

    
    # MINIMUM NORM
    # mne.minimum_norm() # MNE, dSPM, sLORETA
    
    # MIXED NORM
    # mne.mixed_norm() # noise_cov, alpha
    
    # BEAMFORMER
    # mne.beamformer.lcmv() # noise_cov
    # mne.beamformer.dics() # noise_csd
    # mne.beamformer.rap_music() # noise_cov
    
    basename, outdir, figdir = get_output_names(evoked)
    outdir = op.join(outdir, 'source_estimates')
    if not op.exists(outdir):
        os.makedirs(outdir)
    evoked = mne.read_evokeds(evoked)
    
    if isinstance(noise_cov, str):
        noise_cov = mne.read_cov(noise_cov)
    if isinstance(fwd, str):
        fwd = mne.read_forward_solution(fwd)

    try:
        assert len(times) is 2
        tidx = evoked[0].time_as_index(times)
    except TypeError:
        times = [times, times]
        tidx = evoked[0].time_as_index(times)
        tidx[1] += 1
    tslice = slice(*tidx)
    tms = np.asarray(times)*1e3
    
    print('Computing inverse solution')
    print('Method : {}'.format(method))
    
    stcs = list()
    for evo in evoked:
        condition = evo.comment
        print('Conditions : {}'.format(condition))
        
        # Plot whitening
        fig = evo.plot_white(noise_cov, show=False)
        fig.set_size_inches(8,10)
        fig.savefig(op.join(figdir, '{}_whitening_{}').format(basename, condition))
        plt.close('all')
        
        # Make an MEG inverse operator
        print("Making inverse operator")
        inv = mne.minimum_norm.make_inverse_operator(evo.info, fwd, noise_cov,
                                                     loose=None, depth=None, fixed=False)
        
        # Estimate SNR
        print("Estimating SNR at {} ms".format(tms))
        snr, snr_est = mne.minimum_norm.estimate_snr(evo, inv)
        snr_estimate = snr[tslice].mean()
        print('Estimated SNR at {} ms is {:.2f}'.format(tms, snr_estimate))
    
        # Apply the operator
        lambda2 = 1/snr_estimate**2
        
        print("Applying inverse operator")
        stc = mne.minimum_norm.apply_inverse(evo, inv, lambda2, method)
        if crop:
            print('Cropping to {}'.format(times))
            stc.crop(*times) # to save space
            
        print("Writing source estimate")
        stc_name = op.join(outdir, '{}_{}'.format(basename, condition))
        stc.save(stc_name)
        stcs.append(stc_name)
        
        #
        #idx = stc.data[:, times_slice].mean(1).argsort()[::-1][:100]
        #plt.figure()
        #plt.plot(stc.times, stc.data[idx].T)
        #plt.show()
        
    return stcs

def form_contrast( condition1, condition2 ):
    
    condition1 = [condition1] if not isinstance(condition1, list) else condition1
    condition2 = [condition2] if not isinstance(condition2, list) else condition2
    
    # Read STC
    cond1 = list()
    for c in condition1:
        if isinstance(c, str):
            cond1.append(mne.read_source_estimate(c))
    cond2 = list()
    for c in condition2:
        if isinstance(c, str):
            cond2.append(mne.read_source_estimate(c))
    
    # Average internally
    
    c1 = np.mean(np.asarray([c.data for c in cond1]),axis=0)
    c2 = np.mean(np.asarray([c.data for c in cond2]),axis=0)
    
    # Form the contrast
    c3 = cond1[0]
    c3._data = c1-c2
    
    return c3

def form_contrast_evoked(evoked1, evoked2, contrast_name=None):

    evoked1 = [evoked1] if not isinstance(evoked1, list) else evoked1
    evoked2 = [evoked2] if not isinstance(evoked2, list) else evoked2
    
    evoked_contrast = evoked1[0]
    evoked_contrast.comment = contrast_name
    evoked_contrast.data = np.mean(np.asarray([e.data for e in evoked1]), axis=0)
    evoked_contrast.data -= np.mean(np.asarray([e.data for e in evoked2]), axis=0)
    
    return evoked_contrast

def familiar_vs_unfamiliar_evoked(evoked):

    evoked_contrast_name = evoked.split('-ave')[0]+'_Familiar_vs_Unfamiliar-ave.fif'
    evoked = mne.read_evokeds(evoked)
    
    familiar = [e for e in evoked if 'Familiar' in e.comment]
    unfamiliar = [e for e in evoked if 'Unfamiliar' in e.comment]
    
    
    evoked_contrast = form_contrast_evoked(familiar, unfamiliar, 'Familiar > Unfamiliar')    
    evoked_contrast.save(evoked_contrast_name)
    
    return evoked_contrast_name

def faces_vs_scrambled_evoked(evoked):

    evoked_contrast_name = evoked.split('-ave')[0]+'_Faces_vs_Scrambled-ave.fif'
    evoked = mne.read_evokeds(evoked)
    
    faces = [e for e in evoked if 'Familiar' in e.comment or 'Unfamiliar' in e.comment]
    scrambled = [e for e in evoked if 'Scrambled' in e.comment]
    
    evoked_contrast = form_contrast_evoked(faces, scrambled, 'Faces > Scrambled')
    evoked_contrast.save(evoked_contrast_name)
    
    return evoked_contrast_name

def faces_vs_scrambled(stcs):
    
    #stcs = glob(op.join(subject_dir, 'MEEG', 'source_estimates', '*eeg*.stc'))
    
    faces = [s for s in stcs if 'Familiar' in s or 'Unfamiliar' in s]
    scrambled = [s for s in stcs if 'Scrambled' in s]
    
    stc = form_contrast(faces, scrambled)
    stc_name = scrambled[0].rstrip('Scrambled')+'Faces_vs_Scrambled'
    stc.save(stc_name)
    
    return stc_name

def familiar_vs_unfamiliar(stcs):
    familiar = [s for s in stcs if 'Familiar' in s]
    unfamiliar = [s for s in stcs if 'Unfamiliar' in s]
    
    stc = form_contrast(familiar, unfamiliar)
    stc_name = familiar[0].rstrip('Familiar')+'Familiar_vs_Unfamiliar'
    stc.save(stc_name)
    
    return stc_name
    


def plot_stc(stc, src, times, vol=None):
    """
    
    src : str | list | mne.SourceSpaces
    
    surface 2x - if two source files are supplied...
    surface 1x - if one source file (surface) is supplied
    points 1x - always unstructured; plot to nifti or vtk
    
    """
    # IO
    basename, outdir, figdir = get_output_names(stc)
    stc = mne.read_source_estimate(stc)
    
    # Times
    tidx = stc.time_as_index(times)
    tms = [np.round(t*1e3).astype(int) for t in times]
    tms = '-'.join([str(t) for t in tms])
    
    
    if type(src) == mne.SourceSpaces:
        vertices, faces = list(), list()
        for s in src:
            vertices.append(s['rr']*1e3)
            faces.append(None)
    
    if isinstance(src, str):
        src = [src]
        
    if type(src) == list:
        vertices, faces = list(), list()
        for s in src:
            v, f = io_stuff.read_surface(s)
            vertices.append(v)
            faces.append(f)
    
    if vol is None:
        if len(vertices) is 1:
            if faces[0] is None:
                # we only have points...
                n = len(stc.data)
                data = stc.data[:, slice(*tidx)].mean(1)
                data = dict(pointdata=dict(stc=data))
                data = [data]
                names = ['']
            else:
                # we have a mesh, points and faces
                n = len(stc.data)
                #assert len(vertices) == len(faces) == 1
                data = stc.data[:, slice(*tidx)].mean(1)
                if len(vertices) == n:
                    data = dict(pointdata=dict(stc=data))
                elif len(faces) == n:
                    data = dict(celldata=dict(stc=data))
                data = [data]
                names = ['']
            
        elif len(vertices) is 2:
            n_lh = len(stc.lh_data)
            n_rh = len(stc.rh_data)
            
            print('Detected two source spaces...')
            #assert len(vertices) == len(faces) == 2
            
            # Mean source strength in time window
            #stcm = stc.mean()
            lh_data = stc.lh_data[:, slice(*tidx) ].mean(1)
            rh_data = stc.rh_data[:, slice(*tidx) ].mean(1)
            
            if len(vertices[0] == n_lh) and len(vertices[1] == n_rh):
                data = [dict(pointdata=dict(stc=lh_data)),
                        dict(pointdata=dict(stc=rh_data))]
            elif len(faces[0] == n_lh) and len(faces[1] == n_rh):
                data = [dict(celldata=dict(stc=lh_data)),
                        dict(celldata=dict(stc=rh_data))]
            names = ['lh', 'rh']
            
        print('Writing to VTK...', end=' ')
        for v,f,d,n in zip(vertices, faces, data, names):  
            vtk = io_stuff.as_vtk(v, f, **d)
            vtk_name = op.join(figdir, basename+'_{}_{}'.format(n, tms))
            vtk.tofile(vtk_name, 'binary')
        print('Done') 
        
    else:
        vol = nib.load(vol)
        aff = vol.affine
        dim = vol.shape
        vox = mne.transforms.apply_trans(np.linalg.inv(aff), vertices[0])
        vox = np.round(vox).astype(int)
        #assert dim[0] >= vox[:,0].max()
        
        data = stc.data[:, slice(*tidx)].mean(1)
        img = np.zeros(dim)
        img[vox[:,0],vox[:,1],vox[:,2]] = data
        img = nib.Nifti1Image(img, aff)
        vol_name = op.join(figdir, basename+'_{}_{}'.format('vol', tms))
        nib.save(img, vol_name)
        
    """    
    vertices, faces = list(), list()
    for s in src:
        v, f = io_stuff.read_surface(s)
        vertices.append(v)
        faces.append(f)
    
    try:
        n_lh = len(stc.lh_data)
        n_rh = len(stc.rh_data)
        
        print('Detected two source spaces...')
        assert len(vertices) == len(faces) == 2
        
        # Mean source strength in time window
        #stcm = stc.mean()
        lh_data = stc.lh_data[:, slice(*tidx) ].mean(1)
        rh_data = stc.rh_data[:, slice(*tidx) ].mean(1)
        
        if len(vertices[0] == n_lh) and len(vertices[1] == n_rh):
            data = [dict(pointdata=dict(stc=lh_data)),
                    dict(pointdata=dict(stc=rh_data))]
        elif len(faces[0] == n_lh) and len(faces[1] == n_rh):
            data = [dict(celldata=dict(stc=lh_data)),
                    dict(celldata=dict(stc=rh_data))]
        names = ['lh', 'rh']
        
    except AttributeError:
        n = len(stc.data)
        
        assert len(vertices) == len(faces) == 1
        
        data = stc.data[:, slice(*tidx)].mean(1)
        
        if len(vertices) == n:
            data = dict(pointdata=dict(stc=data))
        elif len(faces) == n:
            data = dict(celldata=dict(stc=data))
        data = [data]
        names = ['']
        
    print('Writing to VTK...', end=' ')
    for v,f,d,n in zip(vertices, faces, data, names):  
        vtk = io_stuff.as_vtk(v, f, **d)
        vtk_name = op.join(outdir, basename+'_{}_{}'.format(n, tms))
        vtk.tofile(vtk_name, 'binary')
    print('Done')    
    """
#%% MaxFilter ...
"""
sss = dict(movement_correction=True,
           st_duration=10)
if sss is not None:
    print("Preparing for SSS")
    try:
        mne.preprocessing.maxwell._check_info(raw.info)
    except RuntimeError as e:
        warn("".join(e.args)+" (disabling SSS correction)")
        sss['movement_correction'] = False
    
    if sss['movement_correction']:
        
        try:
            pos = mne.chpi._calculate_chpi_positions(raw, t_step_min=1, t_step_max=1)
        except RuntimeError as e:
            warn("".join(e.args)+" (disabling movement correction by SSS)")
            pos = None
    
    mne.channels.fix_mag_coil_types(raw.info)
    
    print("Remove cHPI noise and line noise from MEG data")
    mne.chpi.filter_chpi(raw)
    
if sss is not None:
    raise ValueError("'sss' not implemented yet")
    
    print("Applying SSS...")
    # bla bla bla something here
    # does it work???
 
    
    pos2 = mne.chpi.read_head_pos("run_01_headpos.txt")
        
    pos = pos[:-2]
    # q1 q2 q3 x y z 
    np.abs(pos[:,1:4] - pos2[:,1:4]).mean()
    np.abs(pos[:,4:7] - pos2[:,4:7]).mean()
    
    
    mne.viz.plot_head_positions(pos)
    mne.viz.plot_head_positions(pos2)
    
    
    # find a close match to st_duration that evenly divides into the experiment
    # duration
    duration = raw.times[-1]-raw.times[0]
    sss['st_duration'] += (duration % sss['st_duration']) / (duration // sss['st_duration'])
    
    # cross_talk=ctc_fname
    # calibration=fine_cal_fname
    
    # tSSS with movement correction
    b = mne.preprocessing.maxwell_filter(raw, st_duration=sss['st_duration'],
                                         head_pos=pos)
    
    # check...
    fmax = 40
    raw.plot_psd(tmin=30,tmax=np.inf,fmax=fmax)
    #a.plot_psd(tmin=30,tmax=np.inf,fmax=fmax)
    #b.plot_psd(tmin=30,tmax=np.inf,fmax=fmax)
    
    plt.figure()
    plt.plot(raw[0][0].T)
    #plt.plot(a[0][0].T)
    plt.plot(b[0][0].T)
    plt.show()
    
    raw.plot()
    b.plot()
"""

"""
if noise_cov is not None:
    if (noise_cov['event_codes'] is not None) and (noise_cov['window'] is not None):
        # useful for resting measurements in MEG or EEG, e.g., with a trigger
        # signifying 'start' of resting block
        
        # noise_cov_window = (0,120) # for 2 min. rest
        # stim_delay = 0
        
        # or noise cov prestimulus
        
        ncw = np.asarray(noise_cov['window']) + stim_delay
          
        print("Computing noise covariance matrix")
        print("Using samples from {} to {} s around triggers".format(*ncw))
        
        events, event_id = compute_stuff.find_events(raw, noise_cov['event_codes'], channels['stim'])
    
        #baseline = (None,None) # use entire range for baseline correction
        
        # Compute noise covariance over all samples of the epoched data
        epo = mne.Epochs(raw, events, event_id, tmin=ncw[0], tmax=ncw[1],
                         baseline=ncw)
        noise_cov = mne.compute_covariance(epo, method="shrunk")
        del epo
        
    else:
        # useful for empty room measurements in MEG
        print("Computing noise covariance matrix")
        print("Using all available samples")
    
        # Compute noise covariance over all samples of the raw data
        noise_cov = mne.compute_raw_covariance(raw, method="shrunk")
        
    noise_cov.save(op.join(save_fif, "{}_noise-cov.fif".format(noise_cov)))  
    
    # Save to PNGs
    figs = noise_cov.plot(raw.info, show=False)
    for fig,which in zip(figs, ["mat", "eig"]):
        fig.set_size_inches(10,10)
        fig.save(op.join(save_fig, "{}_noise-cov_{}.png".format(raw_base, which)))
"""