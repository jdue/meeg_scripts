"""
Imports
"""

from autoreject import (compute_thresholds, LocalAutoRejectCV)
from datetime import datetime
    
from functools import partial
from glob import glob
import matplotlib.pyplot as plt
import mne
from mne.simulation import simulate_evoked, simulate_sparse_stc

from mne.io.constants import FIFF
from numbers import Integral
import numpy as np
import os
import os.path as op
import scipy.sparse as ss
from scipy.spatial import cKDTree
import shutil
from warnings import warn

#import sys
#sys.path.append("C:\\Users\\jdue\\Google Drive\\PYTHON\\packages\\meeg_tools")

from meeg_tools.io import io_misc
from meeg_tools.utils import compute_misc
from meeg_tools.viz import visualize_misc

from sklearn.model_selection import KFold
from scipy.stats import sem

#mne.set_log_level(verbose='WARNING')

#import matplotlib as mpl
#mpl.use('Qt4Agg')

#plt.rcParams['savefig.dpi'] = 600
# =============================================================================

#import configparser
#config = configparser.ConfigParser()
#config.read(sd, 'config.ini')

# HIGH LEVEL FUNCTIONS
# =============================================================================

def prepare_raw(sd, config, i=None):
    """Preprocess raw data.
    """
    if i is None:
        i = 0
        
    run = getf(getd(sd, 'runs'), config['filename'])[i]
    del config['filename'] # not an input to preprocess_raw
    print('Preprocessing run {:d}'.format(i+1))
    raw = preprocess_raw(run, **config)
        
    return raw

def prepare_epochs(sd, config):
    """Epoch data.
    """
    #config['PREPROC_EPOCHS']
    
    raw = getf(getd(sd, 'runs'), '{}*_raw.fif'.format(config['prefix']))
    del config['prefix']
    
    print('Preparing epochs from raw')
    epochs = [preprocess_epochs(r, **config) for r in raw]
    
    return epochs
    

def prepare_concatenated_epochs(sd, config):
    """Concatenate epoched data across runs.
    """
    d = getd(sd)
    epochs = getf(d['runs'], '{}*-epo.fif'.format(config['prefix']))
    
    print('Concatenating epochs')
    epochs = concat_epochs(epochs, d['run_concat'])
    
    return epochs

def prepare_emptyroom_cov(sd, config):
    """Prepare (noise) covariance estimates from empty room measurements.
    """
    
    d = getd(sd)
    emptyroom = config['PATH']['emptyroom']
    #emptyroom = [op.join(emptyroom, i) for i in ['090421', '090707', '090430']]
    #emptyroom = glob(op.join(emptyroom, 6*'[0-9]'+'.fif')) # ugly, I know
    emptyroom = glob(op.join(emptyroom, 6*'[0-9]')) # ugly, I know
    emptyroom = getf(emptyroom, 6*'[0-9]'+'.fif')
    raw = getf(d['runs'][0], '*fmeeg_sss_raw.fif')[0]
    
    
    #raw_name = op.splitext(op.basename(raw.filenames[0]))[0]
    #raw_name += '_sss_raw.fif'
    #raw_name = op.join()
    #raw.save(raw_name)
    
    emptyroom = match_raw_and_emptyroom(emptyroom, raw)
    
    # Apply filter with same settings as for the functional data
    cov = emptyroom_cov(emptyroom, config['PREPROC_RAW']['filt'], d['cov'])
    
    return cov

def preproc_autoreject(sd, config):
    """
    """
    epochs = getf(getd(sd, 'run_concat'), '{}*-epo.fif'.format(config['prefix']))[0]
    del config['prefix']
    
    ar = preprocess_autoreject(epochs, **config)
    
    return ar

def prepare_epochs_cov(sd, config):
    
    d = getd(sd)
    epochs = getf(d['run_concat'], '{}*-epo.fif'.format(config['prefix']))
    
    print('Preparing {} covariance matrix'.format(config['cov_type']))
    cov = [cov_epochs(e, config['cov_type'], d['cov']) for e in epochs]
    
    return cov

def preproc_xdawn(sd, config):
    """Preprocess epochs using XDAWN.
    """
    d = getd(sd)
    epochs = getf(d['run_concat'], '{}*-epo.fif'.format(config['prefix']))[0]
    signal_cov = getf(d['cov'], '{}*signal-cov.fif'.format(config['prefix']))[0]
    del config['prefix']
    
    epochs = preprocess_xdawn(epochs, signal_cov, **config)
    
    return epochs

# LOW LEVEL FUNCTIONS
# =============================================================================

#def cleanup(sd):
#    
#    f = getf(getd(sd, 'runs'), '*')

def preprocess_raw(raw, channels=None, filt=None, phys='ica-reg'):
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
    assert isinstance(raw, str) and op.isfile(raw)
    basename, outdir, figdir = get_output_names(raw)
    
    raw = mne.io.read_raw_fif(raw, preload=True)
    
    if channels is not None:
        print("Relabelling channels and types")
        if 'rename' in channels:
            raw.rename_channels(channels['rename'])
        if 'change_types' in channels:
            raw.set_channel_types(channels['change_types'])
            
        # Set bad channels
        if 'bads' in channels:
            raw.info["bads"] += channels['bads']
            if any(raw.info["bads"]):
                print("The following channels has been marked as bad:")
                print(raw.info["bads"])
    
    
    # Get indices for different channel types (EEG, MAG, GRAD)
    chs = compute_misc.pick_func_channels(raw.info)
    
    # EOG and ECG
    eog = mne.pick_types(raw.info, meg=False, eog=True, ref_meg=False)
    if any(eog):
        veog = mne.pick_channels(raw.info["ch_names"], ["VEOG"])
        heog = mne.pick_channels(raw.info["ch_names"], ["HEOG"])
    ecg = mne.pick_types(raw.info, meg=False, ecg=True, ref_meg=False)
    

    # Maxwell Filtering
    # =========================================================================
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
            
    
    # Squid Jumps
    # =========================================================================
    
    #### Squid jumps ####
    
    # Annotate bad segments inplace
    print("Detecting squid jumps / high frequency stuff...")
    try:
        na = len(raw.annotations)
    except TypeError:
        na = 0
        ##################### HANDLE THIS #####################
        #raise ValueError("handle this -> if no annotations exist")
           
    compute_misc.detect_squid_jumps(raw)
    try:
        print("Annotated {} segments as bad".format(len(raw.annotations)-na))
    except TypeError:
        # no annotations
        print("Annotated 0 segments as bad")

    # Eyeblinks and heartbeats
    # =========================================================================
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
    # Filtering
    # =========================================================================
    
    alpha_level = 0.5
    try:
        tmin = 0+chpi_duration[0] # exclude the initial data annotated bad due to cHPI
    except NameError: # data was not cHPI annotated
        tmin = 0
    tmax = np.inf
    
    try:
        chpi_freqs = [hpi["coil_freq"][0] for hpi in raw.info["hpi_meas"][0]["hpi_coils"]]
    except IndexError:
        # no cHPI info
        chpi_freqs = []
        
    psd_kwargs = dict(average=False, line_alpha=alpha_level, show=False, verbose=False)
    
    # PSDs before filtering
    fig = raw.plot_psd(tmin, tmax, **psd_kwargs)
    fig.savefig(op.join(figdir, "PSD_0_NoFilter_0-nyquist.pdf"))
    fig = raw.plot_psd(tmin, tmax, fmin=0, fmax=50, **psd_kwargs)
    fig.savefig(op.join(figdir, "PSD_0_NoFilter_0-50.pdf"))
    
    if any(chpi_freqs) and 'eeg' in chs:
        # cHPI notch filtering
        #print("Removing cHPI noise at {} Hz from EEG channels".format(chpi_freqs))
        raw.notch_filter(chpi_freqs,
                         picks=np.concatenate((chs["eeg"], eog, ecg)),
                         fir_design='firwin')
    
    # Notch filtering
    if filt['fnotch'] is not None:
        print("Removing line noise at {} Hz".format(filt['fnotch']))
        raw.notch_filter(filt['fnotch'], fir_design='firwin')
        if any(eog) or any(ecg):
            raw.notch_filter(filt['fnotch'], picks=np.concatenate((eog, ecg)),
                             fir_design='firwin')
    
    # 20, 26, 60 ???
    
    fig = raw.plot_psd(tmin, tmax, **psd_kwargs)
    fig.savefig(op.join(figdir, "PSD_1_Notch_0-nyquist.pdf"))
    
    # Highpass filtering
    if filt['fmin'] is not None:
        print("Highpass filtering at {} Hz".format(filt['fmin']))
        raw.filter(l_freq=filt['fmin'], h_freq=None, fir_design='firwin')
        if any(eog) or any(ecg):
            raw.filter(l_freq=filt['fmin'], h_freq=None, picks=np.concatenate((eog, ecg)),
                       fir_design='firwin')
        
        fig = raw.plot_psd(tmin, tmax, fmin=0, fmax=50, **psd_kwargs)
        fig.savefig(op.join(figdir, "PSD_2_NotchHigh_0-50.pdf"))
    
    # Lowpass filtering
    if filt['fmax'] is not None:
        print("Lowpass filtering at {} Hz".format(filt['fmax']))
        raw.filter(l_freq=None, h_freq=filt['fmax'], fir_design='firwin')
        if any(eog) or any(ecg):
            raw.filter(l_freq=None, h_freq=filt['fmax'], picks=np.concatenate((eog, ecg)),
                       fir_design='firwin')
        
        fig = raw.plot_psd(tmin, tmax, **psd_kwargs)
        fig.savefig(op.join(figdir,"PSD_3_NotchHighLow_0-nyquist.pdf"))
    plt.close("all")
    
    basename = 'f'+basename
    
    if 'eeg' in chs:
        print("Rereferencing EEG data to average reference")
        print("(not applying)")
        #raw = mne.add_reference_channels(raw, "REF")
        #raw.set_eeg_reference(projection=False)
        raw.set_eeg_reference(projection=True)#.apply_proj()
        
    # Physiological noise correction
    # =========================================================================
    if (phys is not None) and (not any(np.concatenate((eog, ecg)))):
        warn("No EOG or ECG channels detected. Skipping ICA and RLS...")
        phys = None
    if phys is not None:
        basename = 'p'+basename
        print("Correcting for physiological artifacts")
    
    if phys == "ica-reg":
        print("Using ICA and RLS")
        for ch, picks in chs.items():
            #if ch == 'eeg':
            #    picks = np.concatenate((chs["eeg"], eog, ecg))
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
            ica.fit(raw, picks=picks, reject_by_annotation=False)
            
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
                    fig = visualize_misc.plot_ica_artifacts(ica, eog_inds, eog_scores, "eog")
                    eog_name = op.join(figdir, "ICA_EOG_{}_components".format(ch))
                    fig.savefig(eog_name, bbox_inches="tight")
                    
                    # Plot how components relate to eyeblinks epochs
                    figs = ica.plot_properties(eyeblinks_epoch, picks=eog_inds,
                                               psd_args={"fmax": 30}, show=False)
            
                    try:
                        figs = iter(figs)
                    except TypeError:
                        figs = iter([figs])    
                    for i,fig in enumerate(figs):
                        fig.savefig(op.join(figdir,"ICA_EOG_{}_component_{}".format(ch, i)))
              
            if any(ecg):        
                ecg_inds, ecg_scores = ica.find_bads_ecg(heartbeats_epoch)
                bad_idx += ecg_inds
                print("Found the following ECG related components: {}".format(ecg_inds))
                
                if any(ecg_inds):
                    fig = visualize_misc.plot_ica_artifacts(ica, ecg_inds, ecg_scores, "ecg")
                    ecg_name = op.join(figdir, "ICA_ECG_{}_components".format(ch))
                    fig.savefig(ecg_name, bbox_inches="tight")
        
                    figs = ica.plot_properties(heartbeats_epoch, picks=ecg_inds,
                                               psd_args={"fmax": 80}, show=False)
                    
                    try: 
                        figs = iter(figs)
                    except TypeError:
                        figs = iter([figs])    
                    for i,fig in enumerate(figs):
                        fig.savefig(op.join(figdir, "ICA_ECG_{}_component_{}".format(ch, i)))
    
            plt.close("all")
            
            bad_idx = np.unique(bad_idx)
                
            # Clean bad ICA components using RLS and reconstruct data from ICA
            print("Constructing sources")
            sources = compute_misc.ica_unmix(ica, raw.get_data(picks))
            
            print("Regressing bad components")
            signal = sources[bad_idx]
            reference = raw.get_data()[np.concatenate((eog, ecg))]
            signal = compute_misc.rls(signal, reference)
            sources[bad_idx] = signal
            
            print("Reconstructing data")
            raw._data[picks] = compute_misc.ica_mix(ica, sources)
            
            
    
    elif phys == "ica":
        raise ValueError("'ica' not implemented yet")
        
    elif phys == "reg":
        print("Using RLS")    
        reference = raw.get_data()[np.concatenate((eog, ecg))]
        raw._data[picks] = compute_misc.rls(raw.get_data()[np.concatenate((chs["meg"], chs["eeg"]))], reference)    
    
    elif phys is None:
        print("Not correcting for physiological artifacts")
    
    else:
        IOError("Unknown argument '{}' for 'phys'.".format(phys))
    
    raw_name = op.join(outdir, "{}.fif".format(basename))
    raw.save(raw_name)
    
    return raw_name

"""
raw = '/mrhome/jesperdn/wakeman_henson_meeg_test/EmptyRoom/090409.fif' # data file
raw = '/mrhome/jesperdn/wakeman_henson_meeg_test/EmptyRoom/090421/090421-noSSS.fif'
raw = io_misc.read_data(raw)
mne.channels.fix_mag_coil_types(raw.info)
cal = '/mrhome/jesperdn/wakeman_henson_meeg_test/EmptyRoom/sss_cal.dat' # calibration file
ctc = '/mrhome/jesperdn/wakeman_henson_meeg_test/EmptyRoom/ct_sparse.fif' # cross-talk compensation file
# st_duration ...

def do_maxwell(raw, cal, ctc, coord_frame='head'):
    
    # check if already maxfiltered...
    try:
        mne.preprocessing.maxwell._check_info(raw.info)
        
    except RuntimeError:
        raise
        

    rawsss = mne.preprocessing.maxwell_filter(raw,coord_frame=coord_frame, calibration=cal, cross_talk=ctc)
    
    return raw

cov = emptyroom_cov(raw, filt, d['cov'])
cov2 = emptyroom_cov(raw2, filt, d['cov'])
covsss = emptyroom_cov(rawsss, filt, d['cov'])
"""
def read_covariance(sd, cov_type='noise'):
    
    d = getd(sd, 'cov')
    covf = getf(d, '*{}-cov.fif'.format(cov_type))
    emptyroom = ['emptyroom' in f for f in covf]
    
    # MEG
    cov
    
    
    return f
    

def match_raw_and_emptyroom(emptyroom, raw):
    
    # Find the empty room measurements closest to the acquisition date and use
    # that
    info = mne.io.read_info(raw)
    raw_date = datetime.fromtimestamp(info['meas_date'][0])
    
    acq = list()
    for i,er in enumerate(emptyroom):
        er = op.basename(er)
        y = int(er[:2])+2000
        m = int(er[2:4])
        d = int(er[4:6])
        er_date = datetime(y,m,d)
        acq.append(np.abs((raw_date-er_date).days))
    fname = emptyroom[np.argmin(acq)]
    return fname

def emptyroom_cov(raw, filt, outdir):
    """
    
    Use the same filt dict used to process the
    functional raw data.
    
    
    
    """
    
    if not op.exists(outdir):
        os.makedirs(outdir)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.mkdir(figdir)
    if isinstance(raw, str):
        raw = mne.io.read_raw_fif(raw, preload=True)
    else:
        raw = raw.load_data()
    
    print("Filtering")
    filt['fnotch'] = [f for f in filt['fnotch'] if f<=raw.info['sfreq']/2]
    print("Removing line noise at {} Hz".format(filt['fnotch']))
    raw.notch_filter(filt['fnotch'], fir_design='firwin')
    
    if filt['fmin'] is not None:
        print("Highpass filtering at {} Hz".format(filt['fmin']))
        raw.filter(l_freq=filt['fmin'], h_freq=None, fir_design='firwin')
    if filt['fmax'] is not None:
        print("Lowpass filtering at {} Hz".format(filt['fmax']))
        raw.filter(l_freq=None, h_freq=filt['fmax'], fir_design='firwin')

    print("Computing covariance matrix")
    noise_cov = mne.compute_raw_covariance(raw, method="shrunk")
    noise_cov_name = op.join(outdir, 'emptyroom_noise-cov.fif')
    noise_cov.save(noise_cov_name)
    
    fig_cov, fig_eig = noise_cov.plot(raw.info, show=False)
    fig_cov.savefig(op.join(figdir, "emptyroom_noise-cov.pdf"))
    fig_eig.savefig(op.join(figdir, "emptyroom_noise-eig.pdf"))
    plt.close('all')
    
    return noise_cov_name




def preprocess_epochs(fname_raw, event_codes, stim_chan=None, stim_delay=0, tmin=-0.2,
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
    events, event_id = compute_misc.find_events(raw, event_codes, stim_chan)
          
    # Compensate for stimulus_delay
    tmin += stim_delay
    tmax += stim_delay
    
    # we want baseline correction to get accurate noise cov estimate
    # this may distort the ERP/ERF though !!
    baseline = (tmin, stim_delay) # use stim_delay since we roll back the evoked axis
    
    print('Stimulus delay is {:0.0f} ms'.format(stim_delay*1e3))
    print('Epoching from {:0.0f} ms to {:0.0f} ms'.format(tmin*1e3,tmax*1e3))
    print('Baseline correction using {:0.0f} ms to {:0.0f} ms'.format(baseline[0]*1e3, baseline[1]*1e3))
    
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
    chs = compute_misc.pick_func_channels(evoked[0].info)
    for ch,picks in chs.items():
        for evo in evoked:
            condition = evo.comment
            
            evo_name = "{}_{}_{}".format(basename, ch, condition)
            evo_name = op.join(figdir, evo_name)
            
            fig = evo.plot(picks=picks, spatial_colors=True, exclude='bads',
                           show=False)
            fig.axes[0].set_title("Evoked '{}' [{}]".format(condition, ch.upper()))
            fig.savefig(evo_name+".pdf")
        plt.close('all')
    
    return epo_name



def concat_epochs(fname_epochs, outdir=None):
    """
    Concatenate list of epochs from filenames.
    
    """
    assert isinstance(fname_epochs, list) and len(fname_epochs) > 1
    
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

def preprocess_autoreject(fname_epochs, stim_delay=0, kappa=None, n_jobs=1):
    """
    
    epochs : dict
    
    obj_dir : str
        Directory in which to save the epochs
        
    
    """
    
    assert isinstance(fname_epochs, str) and op.isfile(fname_epochs)
    basename, outdir, figdir = get_output_names(fname_epochs)
    basename = 'a'+basename
    
    epochs = mne.read_epochs(fname_epochs)
    ntrials = len(epochs)    
    nchan = len(mne.pick_types(epochs.info, meg=True, eeg=True, ref_meg=False))
    
    # Parameter space to use for cross validation
    # rho   : the maxmimum number of "bad" channels to interpolate
    # kappa : fraction of channels which has to be deemed bad for an epoch to
    #         be dropped.
    if kappa is None:
        kappa = np.linspace(0, 1.0, 11)
    rho = np.round(np.arange(nchan/20, nchan/4, nchan/20)).astype(np.int)
    
    # Args for threshold detection function
    thresh_func = partial(compute_thresholds, method="random_search",
                          n_jobs=n_jobs)
    
    
    print("Running AutoReject")
    print("------------------")
    print('# jobs   : {}'.format(n_jobs))
    print("# trials : {}".format(ntrials))
    print("kappa    : {}".format(kappa))
    print("rho      : {}\n".format(rho))
    
    # Fit (local) AutoReject using CV
    ar = LocalAutoRejectCV(rho, kappa, thresh_func=thresh_func)

    print('Fitting parameters')
    ar = ar.fit(epochs)
    print('Transforming epochs')
    clean_epochs = ar.transform(epochs)
    
    # Save
    epochs_name =  op.join(outdir, "{}-epo.fif".format(basename))
    clean_epochs.save(epochs_name)

    
    # Evoked response after AR
    clean_evoked = list()
    for condition in clean_epochs.event_id.keys():
        evo = clean_epochs[condition].average()
        evo.shift_time(-stim_delay)
        clean_evoked.append(evo)

    evo_name = op.join(outdir, "{}-ave.fif".format(basename))
    mne.evoked.write_evokeds(evo_name, clean_evoked)
    
    # Visualize results of autoreject
    
    # Bad segments
    fig_bad, fig_frac = visualize_misc.viz_ar_bads(ar)
    fig_name = op.join(figdir, "Epochs_bad_segments.pdf")
    fig_bad.savefig(fig_name, bbox_inches="tight")
    fig_name = op.join(figdir, "Epochs_bad_fractions.pdf")
    fig_frac.savefig(fig_name, bbox_inches="tight")
    
    # Check for bad channels
    bad_cutoff = 0.5
    bad_frac = ar.bad_segments.mean(0)
    possible_bads = [epochs.ch_names[bad] for bad in np.where(bad_frac>bad_cutoff)[0]]
    
    for cevo in clean_evoked:                
        condition = cevo.comment
        
        evo_name = op.join(figdir, "{}_{}".format(basename, condition))
        
        fig = cevo.plot(spatial_colors=True, exclude='bads', show=False)
        fig.axes[0].set_title("Evoked Response '{}'".format(condition))
        fig.savefig(evo_name+".pdf")
    
        if any(possible_bads):
            # Evoked response before AR
            evo = epochs[condition].average()
            evo.shift_time(-stim_delay)
            evo.plot(spatial_colors=True, exclude='bads', show=False)
            
            # Plot the bad channels before and after AR
            tb = evo.copy()
            ta = cevo.copy()
            tb.info["bads"] = possible_bads
            ta.info["bads"] = possible_bads
            
            fig = tb.plot(spatial_colors=False, exclude=[], show=False)
            fig.axes[0].set_title("Evoked Response '{}' before AutoReject".format(condition))
            fig.savefig(evo_name+"-bads-before.pdf")
            fig = ta.plot(spatial_colors=False, exclude=[], show=False)
            fig.axes[0].set_title("Evoked Reponse '{}' after AutoReject".format(condition))
            fig.savefig(evo_name+"-bads-after.pdf")
    
        plt.close('all')
        
    return epochs_name
    
    """
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
        #cepo = ar.fit_transform(epo)
        ar = ar.fit(epo)
        cepo = ar.transform(epo)
        
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
        fig_bad, fig_frac = visualize_misc.viz_ar_bads(ar)
        fig_name = op.join(figdir, "Epochs_bad_segments_{}.pdf".format(ch))
        fig_bad.savefig(fig_name, bbox_inches="tight")
        fig_name = op.join(figdir, "Epochs_bad_fractions_{}.pdf".format(ch))
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
            fig.savefig(evo_name+".pdf")
        
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
                fig.savefig(evo_name+"-bads-before.pdf")
                fig = ta.plot(spatial_colors=False, exclude=[], show=False)
                fig.axes[0].set_title("Evoked '{}' [{}] after AutoReject".format(condition, ch.upper()))
                fig.savefig(evo_name+"-bads-after.pdf")
        
        plt.close('all')
    
    return epochs"""

def cov_epochs(epochs, cov_type='noise', outdir=None):
    """
    Noise or signal covariance from epochs.
    """

    assert isinstance(epochs, str) and op.isfile(epochs)
    basename, outdir2, figdir = get_output_names(epochs)
    if outdir is None:
        outdir = outdir2
    if not op.exists(outdir):
        os.makedirs(outdir)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.mkdir(figdir)
    
    epochs = mne.read_epochs(epochs)

    # Covariance window
    if cov_type == 'noise':
        win = epochs.baseline
    elif cov_type == 'signal':
        win = (None, None)
    else:
        raise TypeError("cov_type must be 'noise' or 'signal'")
    
    print('Using time window {} to {}'.format(*win))
    cov = mne.compute_covariance(epochs, tmin=win[0], tmax=win[1],
                                 method="shrunk")
    cov_name = op.join(outdir, "{}_{}-cov.fif".format(basename, cov_type))
    cov.save(cov_name)
    
    fig_cov, fig_eig = cov.plot(epochs.info, show=False)
    fig_cov.savefig(op.join(figdir, "{}_{}-cov.pdf".format(basename, cov_type)))
    fig_eig.savefig(op.join(figdir, "{}_{}-eig.pdf".format(basename, cov_type)))
    plt.close('all')
    
    return cov_name
    
def preprocess_xdawn(epochs, signal_cov, stim_delay=0):
    """
    apply xdawn to each channel type separately...
    
    """
    
    assert isinstance(epochs, str) and op.isfile(epochs)
    basename, outdir, figdir = get_output_names(epochs)
    basename = 'x'+basename
    
    epochs = mne.read_epochs(epochs)
    signal_cov = mne.read_cov(signal_cov)
    
    # Pick only functional types so as to match signal covariance data
    epochs.pick_types(meg=True, eeg=True, ref_meg=False)
    
    #scale = dict(eeg=1e6, grad=1e13, mag=1e15)
    
    
    chs = compute_misc.pick_func_channels(epochs.info)
    for ch, picks in chs.items():
        print('Processing', ch.upper())
        epo = epochs.copy()
        ch_names = [epochs.ch_names[i] for i in picks]
        epo.pick_channels(ch_names)
        signal_cov_data = signal_cov.data[picks][:,picks]
        
        print('Denoising using xDAWN')
        print('Estimating parameters using cross validation')
        n_components, fig = xdawn_cv(epo, signal_cov_data)
        fig.set_size_inches(10,5)
        fig.savefig(op.join(figdir, '{}_{}_xdawn_cv.pdf').format(basename, ch))
    
        print('Applying xDAWN')
        data = np.zeros_like(epo.get_data())
        for condition, eidv in epo.event_id.items():
            # Fit each condition separately
            print(condition)
            xd = mne.preprocessing.Xdawn(n_components[condition],
                                         signal_cov_data,
                                         correct_overlap=False)
            xd.fit(epo[condition])
            x = xd.apply(epo[condition])[condition].get_data()
            data[epo.events[:,2]==eidv] = x
            
            # Component time series
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot((epo.times-stim_delay)*1e3,
                    xd.transform(epo[condition].get_data()).mean(0).T)#*scale[ch])
            ax.set_xlabel('Time (ms)')
            plt.legend([str(i) for i in range(xd.n_components)])
            ax.set_title("xDAWN time courses '{}'".format(condition))
            fig.savefig(op.join(figdir, '{}_{}_xdawn_timecourses_{}.pdf'.format(basename, ch, condition)))
            plt.close('all')
            
            # Component patterns
            nrows = 2
            ncols = 4
            fig, axes = plt.subplots(2, 4)
            i = 0
            for row in range(nrows):
                for col in range(ncols):
                    mne.viz.plot_topomap(xd.patterns_[condition].T[i], epo.info,
                                         axes=axes[row,col], show=False)
                    i += 1
            fig.suptitle("xDAWN patterns '{}' [{}]".format(condition, ch))
            fig.tight_layout()
            fig.savefig(op.join(figdir, '{}_{}_xdawn_patterns_{}.pdf'.format(basename, ch, condition)))
            plt.close('all')
            
        # Update original epochs object
        epochs._data[:, picks, :] = data.copy()
        
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
        #fig.axes[0].set_title("Evoked '{}'".format(condition))
        fig.savefig(evo_name+".pdf")
            
    plt.close('all')
        
    return epo_name

def xdawn_cv(epochs, signal_cov, component_grid=None, nfolds=5, cv_curve=True):
    """Use cross validation to find the optimal number of xDAWN components to
    project the epochs onto.
    
    epochs : MNE.Epochs object
    
    signal_cov : 
        
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
            print('Fold', i+1, 'of', nfolds)
            for j, n_components in enumerate(component_grid):
                train_set = epo.copy().drop(test)
                test_set = epo.copy().drop(train)
                
                xd = mne.preprocessing.Xdawn(n_components, signal_cov, correct_overlap=False)
                xd.fit(train_set)
                # shouldn't xd be applied to the test set???
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
    
    
def prepare_contrasts(sd, config):
    
    if isinstance(config['contrasts'], dict):
        config['contrasts'] = [config['contrasts']]
        
    d = getd(sd)
    evoked = getf(d['run_concat'], 'apf*-ave.fif')
    
    print('Preparing {} contrast(s)'.format(len(config['contrasts'])))
    cevo = []
    for c in config['contrasts']:
        c['outdir'] = d['contrasts']
        for evo in evoked:
            cevo.append(make_contrast(evo, **c))
        
    
def make_contrast(evoked, c1=None, c2=None, c1_name=None, c2_name=None,
                  outdir=None):
    """
    does the contrast c1 > c2 and saves the result.
    
    PARAMETERS
    ----------
    evoked : str
        Filename of the evoked data from which to form contrasts.
    c1 : str | list of str
        (Exact) name of contrast(s).
    c1 : str | list of str
        (Exact) name of contrast(s).
    c1_name : str
        Name corresponding to c1. Mandatory if len(c1) > 1.
    c2_name : str
        Name corresponding to c1. Mandatory if len(c1) > 1.
    """
    if c1 is None and c2 is None:
        raise ValueError
        
    if outdir is None:
        outdir = op.dirname(evoked)
    
    if c1:
        c1 = [c1] if not isinstance(c1, list) else c1
    if c2:
        c2 = [c2] if not isinstance(c2, list) else c2
    
    if c1 and c1_name is None:
        assert len(c1) is 1, 'Cannot auto-name contrasts with multiple conditions'
        c1_name = c1[0]
    if c2 and c2_name is None:
        assert len(c2) is 1, 'Cannot auto-name contrasts with multiple conditions'
        c2_name = c2[0]
    
    if c1 is not None and c2 is not None:    
        cfname = c1_name+'_vs_'+c2_name
        ccname = c1_name+' > '+c2_name
    elif c1:
        cfname = ccname = c1_name
    elif c2:
        cfname = ccname = c2_name
    cfname = cfname.replace(' ', '_')
        
    # Outputs
    base = op.splitext(op.basename(evoked))[0]
    outdir = op.join(outdir, cfname)
    if not op.exists(outdir):
        os.makedirs(outdir)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.makedirs(figdir)
    
    evoked = mne.read_evokeds(evoked)
    
    # Get the contrasts
    ename = [e.comment for e in evoked]
    weights = np.zeros(len(evoked))
    if c1:
        weights[[ename.index(c) for c in c1]] = 1/len(c1)
    if c2:
        weights[[ename.index(c) for c in c2]] = -1/len(c2)
               
    #e1 = [e for c in c1 for e in evoked if e.comment == c]
    #e2 = [e for c in c2 for e in evoked if e.comment == c]
    
    contrast = mne.combine_evoked(evoked, weights)
    contrast.comment = ccname
    contrast.save(op.join(outdir, base+'.fif'))
    
    # Make the new evoked object
    #ec = e1[0]
    #ec.comment = ccname
    #ec.data = np.mean(np.asarray([e.data for e in e1]), axis=0)
    #ec.data -= np.mean(np.asarray([e.data for e in e2]), axis=0)
    #ec.save(op.join(outdir, base+'.fif'))
    
    # Plot

    fig = contrast.plot(spatial_colors=True, exclude='bads', show=False)
    #fig.axes[0].set_title("Evoked '{}' [{}]".format(ec.comment, ch.upper()))
    #if len(chs) > 1:
    #    fig.savefig(evo_cname+'_{}_.pdf'.format(ch))
    #else:
    fig.savefig(op.join(figdir, base+".pdf"))
    plt.close()
    
    return contrast
    


def get_output_names(fname_raw):
    
    outdir = op.dirname(fname_raw)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.mkdir(figdir)
    basename, _ = op.splitext(op.basename(fname_raw)) 
    basename = basename.split('-')[0]
    
    return basename, outdir, figdir

def initialize_structure(sd):
    
    d = getd(sd)
    
    # Check for existance
    for _,v in d.items():
        if isinstance(v, list):
            for vv in v:
                if isinstance(vv, str) and not op.exists(vv):
                    os.makedirs(vv)
        else:
            if not op.exists(v):
                os.makedirs(v)

def getd(sd, k=None):
    """Get the different directories...
    if k is specified return only that directory in which case the result is a
    string - otherwise a dict
    """
    
    d = dict()
    d['meeg'] = op.join(sd, 'MEEG')
    d['cov'] = op.join(d['meeg'], 'cov')  
    d['runs'] = sorted(glob(op.join(d['meeg'], 'run_[0-9]*')),key=str.lower)
    d['run_concat'] = op.join(d['meeg'], 'run_concat')  
    d['fwd'] = op.join(d['meeg'], 'forward')  
    d['fwds'] = sorted(glob(op.join(d['fwd'], '*')), key=str.lower)  
    d['inv'] = op.join(d['meeg'], 'inverse')
    d['contrasts'] = op.join(d['meeg'], 'contrasts') 
    d['precomp'] = op.join(d['meeg'], 'precomputed')  
    d['smri'] = op.join(sd, 'sMRI')
          
    if k is not None:
        d = d[k]
    
    return d

def getf(path, pattern, recursive=False):
    """Recursive search for files matching 'pattern' in directory/ies 'sd'.
    """
    
    if isinstance(path, str):
        if recursive:
            f = glob(op.join(path, '**', pattern), recursive=True)
        else:
            f = glob(op.join(path, pattern))
    elif isinstance(path, list):
        f = list()
        for p in path:
            if recursive: 
                ff = glob(op.join(p,'**', pattern), recursive=True)
            else:
                ff = glob(op.join(p, pattern))
            try:
                f.append(ff[0])
            except IndexError:
                pass
            #f = [try: glob(op.join(p,'**', pattern), recursive=True)[0] for p in path]
            
    if not any(f):
        raise IOError('No files matching {:s} in {:s}'.format(pattern, path))
    
    return sorted(f, key=str.lower)

def get_surfs(sd, config):
    """
    """
    df = getfwdd(sd, config)
    hm = df['hm']
    #hm = getd(sd, 'hm')
    
    surfs = getf(hm, '*.stl')
    keys = ['air', 'bone', 'csf', 'eyes', 'gm', 'skin',
                             'ventricles', 'wm']
    surfs = {k:s for k in keys for s in surfs if k in op.basename(s)}
    return surfs

def load_mrifids(mrifids):
    fids_mri_coords = np.loadtxt(mrifids, usecols=(0,1,2))
    fids_mri_coords /= 1000 # convert to m
    fids_mri_labels = np.loadtxt(mrifids, dtype="S3", usecols=3)
    fids_mri = dict()
    for i,ii in zip(fids_mri_labels, fids_mri_coords):    
        fids_mri[str(i,"utf-8").lower()] = ii
    return fids_mri

def getfwdd(sd, config):
    """
    Setup directories for forward solution.
    """
    d = getd(sd)
    fwd = op.join(d['fwd'], config['name'])
    
    df = dict()    
    for k in ['hm', 'coreg', 'gain', 'src']:
        df[k] = op.join(fwd, k)
        if not op.exists(df[k]):
            os.makedirs(df[k])
    return df
    
def prepare_for_forward(sd, config):
    """
    
    Move surface files
    Move source space files
    Move gain matrix
        
    """
    d = getd(sd)
    df = getfwdd(sd, config)
    
    # If there are no headmodel files, move if possible
    if len(glob(op.join(df['hm'],'*'))) == 0:
        # no headmodel files try to copy from hm_files...
        surfs = glob(op.join(d['precomp'], config['name'], 'surfs','*'))
        if len(surfs) == 0:
            surfs = glob(op.join(d['precomp'], 'surfs', '*'))
        for s in surfs:
            dst = op.join(df['hm'], op.basename(s))
            print('Copying {} to {}'.format(s,dst))
            shutil.copy(s, dst)
    else:
        print('Head model surfaces already present.')
    
    # If there are no source space files, move if possible
    if len(glob(op.join(df['src'], '*'))) == 0:
        # no source space files
        src = glob(op.join(d['precomp'] , config['name'], 'src','*'))
        if len(src) == 0:
            src = glob(op.join(d['precomp'], 'src', '*'))
        for s in src:
            d = op.join(df['src'], op.basename(s))
            print('Copying {} to {}'.format(s,d))
            shutil.copy(s, d)
    else:
        print('Source space files already present.')
    
    if config['precomp']:
        # Copy precomputed gain matrix
        srcgain = op.join(d['precomp'], config['name'], 'gain')
        dstgain = glob(op.join(df['gain'], '*-fwd.fif'))
        
        if op.exists(srcgain) and len(dstgain) is 0:
            # the the precomputed leadfield...
            srcgain = glob(op.join(srcgain, '*-fwd.fif'))
            assert len(srcgain) is 1
            srcgain = srcgain[0]
            
            print('Copying precomputed gain matrix from {} to {}'.format(srcgain, dstgain))
            dstgain = op.join(df['gain'], op.basename(srcgain))
            shutil.copy(srcgain, dstgain) # or move..?
    
    return df
    

def coreg(sd, config):
    """
    
    sd : str
    
    """
    d = getd(sd)
    df = getfwdd(sd, config)
    
    # Get files
    raw = getf(d['runs'][0], '*fmeeg_sss_raw.fif')[0]
    info = mne.io.read_info(raw)
    mrifids = op.join(d['smri'], 'mri_fids.txt')
    skin = get_surfs(sd, config)['skin']
    
    # Run
    trans = coregister(info, mrifids, skin, df['coreg'])
    
    return trans

def coregister(info, mrifids, skin, coregdir):
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
    
    
    # Fiducials MRI
    fids_mri = load_mrifids(mrifids)
    
    # Skin surface
    vertices, faces = io_misc.read_surface(skin)
    vertices /= 1000 
    scalp = dict(v=vertices, f=faces)
    
    print('Coregistering MRI to head coordinates')
    # Get the transformation
    trans, eeg_locs_proj, digs_proj = compute_misc.get_mri2head_transform(info, fids_mri, scalp,
                                                 rm_points_below_nas=True)
    
    trans_name = op.join(coregdir, 'mri_head-trans.fif')
    trans.save(trans_name)

    # Transform from MRI to head coordinates
    # Convert back to mm and write
    
    # MRI fiducials in head coordinates
    mri_fids_base = op.basename(mrifids)
    mri_fids_base, _ = op.splitext(mri_fids_base)
    fids_mri = np.asarray([v for v in fids_mri.values()])
    mri_fids_head = mne.transforms.apply_trans(trans, fids_mri)
    vtk = io_misc.as_vtk(mri_fids_head*1e3)
    vtk.tofile(op.join(coregdir, 'headcoord_'+mri_fids_base))
    
    # Scalp mesh in head coordinates
    scalp_base = op.basename(skin)
    scalp_base, _ = op.splitext(scalp_base)
    scalp['v'] = mne.transforms.apply_trans(trans, scalp['v'])
    vtk = io_misc.as_vtk(scalp['v']*1e3, scalp['f'])
    vtk.tofile(op.join(coregdir, 'headcoord_'+scalp_base))
    
    # Sensors
    locs = compute_misc.get_sensor_locs(info)
    for ch, loc in locs.items():
        vtk = io_misc.as_vtk(loc*1e3)
        vtk.tofile(op.join(coregdir, 'headcoord_{}_locs'.format(ch)))
        
    # Digitized points
    digs, _, fids = compute_misc.get_digitized_points(info)
    vtk = io_misc.as_vtk(digs*1e3)
    vtk.tofile(op.join(coregdir, 'headcoord_digs'))
    vtk = io_misc.as_vtk(np.asarray([v for v in fids.values()])*1e3)
    vtk.tofile(op.join(coregdir, 'headcoord_meg_fids'))

    # EEG electrodes projected onto scalp
    vtk = io_misc.as_vtk(eeg_locs_proj*1e3)
    vtk.tofile(op.join(coregdir, 'headcoord_eeg_locs_proj'))

    # Digitized points projected onto scalp
    vtk = io_misc.as_vtk(digs_proj*1e3)
    vtk.tofile(op.join(coregdir, 'headcoord_digs_proj'))
    
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

def prepare_sourcespace(sd, config):
    
    
    #print('Preparing source space...')
    df = getfwdd(sd, config)
    src = getf(df['src'], '[lr]*')
    
    src = io_misc.sourcespace_from_files(src)
    src_name = op.join(df['src'], 'sourcespace-src.fif')
    src.save(src_name, overwrite=True)
    
    return src
    

def prepare_bem(sd, config):
    """
    """
    df = getfwdd(sd, config)
    surfs = get_surfs(sd, config)
    
    bem = prepare_bem_mne(surfs['csf'], surfs['bone'], surfs['skin'], df['hm'],
                          config['conductivity'])
    
    return bem

def prepare_bem_mne(brain, skull, scalp, outdir, conductivity=(0.3, 0.006, 0.3)):
    """Compute BEM model using MNE.    
    """
    
    
    
    # Setup some constants
    ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN,
           FIFF.FIFFV_BEM_SURF_ID_SKULL,
           FIFF.FIFFV_BEM_SURF_ID_HEAD]
    
    # Read surfaces for BEM
    surfs = []
    for surf in (brain, skull, scalp):
        v,f = io_misc.read_surface(surf)
        surfs.append(dict(rr=v, tris=f, ntri=len(f), use_tris=f, np=len(v)))
    
    
    print('Preparing BEM model...')
    bem = mne.bem._surfaces_to_bem(surfs, ids, conductivity)
    print('Writing BEM model...')
    bem_name = op.join(outdir, "bem-surfs.fif")
    mne.write_bem_surfaces(bem_name, bem)
    
    print('Computing BEM solution...')
    bem = mne.make_bem_solution(bem)
    print('Writing BEM solution...')
    bemsol_name = op.join(outdir, "bem-sol.fif")
    mne.write_bem_solution(bemsol_name, bem)
    print('Done')
    
    return bem

    
def prepare_forward(sd, config):
    
    d = getd(sd, 'runs')
    df = getfwdd(sd, config)
    
    raw = getf(d[0], '*fmeeg_sss_raw.fif')[0]
    src = getf(df['src'], '*-src.fif')[0]
    bem = getf(df['hm'], '*bem-sol.fif')[0]
    trans = getf(df['coreg'], '*-trans.fif')[0]
    
    fwd = prepare_forward_mne(raw, src, bem, trans, df['gain'])
    
    return fwd

def prepare_forward_mne(raw, src, bem, trans, outdir):    
    """
    
    """
    
    if isinstance(raw, str):
        raw = io_misc.read_data(raw)
    if isinstance(bem, str):
        bem = mne.read_bem_solution(bem)
    if isinstance(src, str):
        src = mne.read_source_spaces(src)
    if isinstance(trans, str):
        trans = mne.read_trans(trans)
    
    # MNE automatically projects EEG electrodes onto the scalp surface
    print('Computing forward solution')
    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=True, mindist=0, n_jobs=1)
    
    print('Writing forward solution')
    fwd_name = op.join(outdir, "forward-fwd.fif")
    mne.write_forward_solution(fwd_name, fwd)
    print('Done')
    
    return fwd


######
# =============================================================================


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

def compute_whitener(inv, return_inverse=False):
    """
    Use mne.cov.compute_whitener to compute W which whitens AND transforms the
    data back to the original space 
    """
    # Whitener
    eig = inv['noise_cov']['eig']
    nonzero = eig > 0
    ieig = np.zeros_like(eig)
    ieig[nonzero] = 1/np.sqrt(eig[nonzero])
    ieig = np.diag(ieig)
    eigvec = inv['noise_cov']['eigvec']
    W = ieig @ eigvec
    #W = eigvec.T @ ieig @ eigvec # what whiten_evoked does; projects it back to original space
    
    iW = eigvec.T @ np.diag(np.sqrt(eig))
    
    # such that iW @ W == I (approx.)
    
    if return_inverse:
        return W, iW
    else:
        return W

def compute_MNE_cost(evoked, inv, lambda2, twin, return_parts=False):
    """Compute the cost function of a minimum norm estimate. The cost function
    is given by
    
        S = E.T @ E + lambda2 * J.T @ R**-1 @ J
        
    where E = X - Xe, i.e., the error between the actual data, X, and the data
    as estimated (predicted) from the inverse solution, lambda2 is the
    regularization coefficient, J is the (full) current estimate (i.e., the
    inverse solution), and R is the source covariance which depends on the
    particular MNE solver.
    
    MNE employs data whitening and SVD of the gain matrix. 
    
    
    
    See also
    
        https://martinos.org/mne/stable/manual/source_localization/inverse.html
        
        
    inv : *Prepared* inverse operator
    
    """
    if twin is not None:
        assert len(twin) is 2
        tidx = evoked.time_as_index(twin)
        X = evoked.data[:,slice(*tidx)]
    else:
        X = evoked.data
    #nave = evoked.nave
    
    # SVD gain matrix
    U = inv['eigen_fields']['data'].T # eigenfields
    V = inv['eigen_leads']['data']    # eigenleads
    S = inv['sing']                   # singular values

    
    #gamma_reg = S / (S**2 + lambda2)  # Inverse, regularized singular values
    gamma_reg = inv['reginv']
    Pi = S * gamma_reg                # In the case of lambda2 == 0, Pi = I
    
    # Source covariance
    R = inv['source_cov']['data'][:,None]
    iR = 1/R
    
    # Whitening operator
    W, iW = compute_whitener(inv, return_inverse=True)
    
    # Inverse operator
    M = np.sqrt(R) * V * gamma_reg[None,:] @ U.T
    
    
    # Current estimate (the inverse solution)
    J = M @ W @ X
       
    # Estimated (predicted) data from the inverse solution. If lambda2 == 0,
    # then this equals the actual data. If lambda2 > 0, there will be some
    # deviation (error) from this.
    Xe = iW @ U * Pi[None,:] @ U.T @ W @ X

    """
    # ALTERNATIVE
    # Calculating the estimated (predicted) data from the (full, i.e., all
    # components, x, y, z) inverse solution is done like so:
    
    # Get the solution ('sol') from mne.minimum_norm.inverse line 795. Don't
    # apply noise_norm to sol. Then undo source covariance (depth) weighting.
    # MNE also weighs by the effective number of averages ('nave') such that
    # R = R / nave that thus iR = 1/R * nave, however, since we are using a
    # *prepared* inverse operator, R has already been scaled by nave so no need
    # to undo this manually.
    J /= np.sqrt(R)
    
    # Calculate the whitened (and weighted) gain matrix
    # Note that the eigenfields are transposed, such that U is in fact
    # inv['eigen_fields']['data'].T. We get    
    G = U * S[None,:] @ V.T
    
    # To be clear,
    #
    #   inv['eigen_fields'] - stores U.T
    #   inv['eigen_leads']  - stores V
    # 
    # such that
    # 
    #     gain = USV.T
    
    # Finally, the estimated (recolored) data is obtained from
    Xe = iW @ G @ J
    """
    #
    # Cost function
    # 
    # The prediction term
    E = X - Xe    # Error
    WE = W @ E    # White error
    #WE = np.sum(WE**2)
    #JP = lambda2 * np.sum(J**2 * iR) # current norm penalty
    
    WE = np.sum(WE**2)
    JP = lambda2 * np.sum(J**2 * iR)
    
    # Cost in 'twin' or over all time points
    cost = WE + JP
    
    """
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(X.T*1e6)
    plt.subplot(3,1,2)
    plt.plot(Xe.T*1e6)
    plt.subplot(3,1,3)
    plt.plot(E.T*1e6)
    """
    if return_parts:
        return cost, WE, JP
    return cost


def get_noise_covs(covd, mods):
    
    # Choose modality specific noise covariance matrices over general
    # Nested list comprehensions...
    noise_cov = []
    noise_cov_all = getf(covd, '*noise-cov.fif', recursive=False)
    if isinstance(noise_cov_all, str):
        noise_cov = [noise_cov_all]*len(mods)
    else:
        noise_cov_empty = [n for n in noise_cov_all if 'emptyroom' in n]
        noise_cov_spe = [n for n in noise_cov_all if not 'emptyroom' in n and any([m for m in mods if m in op.basename(n)])]
        #noise_cov_gen = [n for n in noise_cov_all if not any([m for m in mods if m in op.basename(n)])]
        noise_cov_gen = getf(covd, '*raw_noise-cov.fif', recursive=False)
        if any(noise_cov_gen) and isinstance(noise_cov_gen, str):
            noise_cov_gen = [noise_cov_gen]
        # We can only have one 'general' noise covariance matrix file
        assert len(noise_cov_gen) in [0, 1], 'Number of general noise covs was {}'.format(len(noise_cov_gen))
        
        for m in mods:
            add = []
            # First, try if there is empty room data for this modality
            add = [n for n in noise_cov_empty if m in op.basename(n)]
            if not any(add):
                # If not, try an other modality specific covariance matrix
                add = [n for n in noise_cov_spe if m in op.basename(n)]
            if not any(add):
                # If not, use the general covariance (which supposedly contains
                # more than one modality)
                add = noise_cov_gen
            noise_cov += add
            
    return noise_cov

def prepare_inverse(sd, config, i=None):
    """
    
    If i == None, prepare inverse solution only for the ith forward model.
    Otherwise, do it for all forward models.
    
    """
    # get the input evoked
    # get the noise covariance
    # get the fwd models for which to compute inverse sol
    
    # which fwd models...
    if i is None:
        i = 0
        
    d = getd(sd)
    if not isinstance(config['contrast'], list):
        config['contrast'] = [config['contrast']]
        
    for contrast in config['contrast']:    
        print('Contrast : {}'.format(contrast))
        
        assert contrast is not None
        cd = op.join(d['contrasts'], contrast)
        if 'simulation' in contrast.lower():
            evoked = getf(cd, 'pf*-ave.fif') # created from raw
        else:
            evoked = getf(cd, '{}*-ave.fif'.format(config['prefix']))
                
        if not any(evoked):
            raise IOError('No evoked data found.')
        #mods = ['_'+op.basename(e).split('-')[0].split('_')[-1]+'_' for e in evoked]
        #noise_cov = get_noise_covs(d['cov'], mods)
        noise_cov = getf(d['cov'], '{}*noise-cov.fif'.format(config['prefix']))
        if not any(noise_cov):
            raise IOError('No covariance data found.')
            
        print('Preparing inverse solution')
        fwdd = d['fwds'][i]
        fwd = getf(op.join(fwdd, 'gain'), '*-fwd.fif')[0]
        fwdb = op.basename(fwdd)
        outdir = op.join(d['inv'], contrast, config['method'], fwdb)
        
        if not any(fwd):
            raise IOError('Gain matrix not found.')
        
        print(op.basename(fwdb))
        for evo, cov in zip(evoked, noise_cov):
            cost = do_inverse(evo, cov, fwd, twin=config['twin'],
                              method=config['method'],
                              fwd_normal=config['fwd_normal'], outdir=outdir)
            
    print('Done')
    
    return cost
    
def plot_inverse(sd, config, i=None):
    
    # name of fwd
    # name of stc
    # times
    if i is None:
        i = 0
    
    if not isinstance(config['contrast'], list):
        config['contrast'] = [config['contrast']]
        
    d = getd(sd)
    
    fwdd= d['fwds'][i]
    fwdb = op.basename(fwdd)
    print(fwdb)
    
    # Source space
    srcd = getfwdd(sd, dict(name=fwdb))['src']
    src = getf(srcd, '[lr]*')
    
    # Source estimate
    for contrast in config['contrast']:
        invd = op.join(d['inv'], contrast, config['method'], fwdb)
        stc = getf(invd, '*stc')
        stc = [s.rstrip('-[lr]h.stc') for s in stc]
        stc = list(set(stc))
        
        for s in stc:
            plot_stc(s, src, config['twin'])
            if config['clean_stc_on_plot']:
                print('Removing source estimates:')
                for f in glob(s+'*'):
                    print(f)
                    os.remove(f)

def do_inverse(evoked, noise_cov, fwd, twin, method='dSPM', signal_cov=None,
               fwd_normal=False,
               crop=False, outdir=None):
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
    basename, _, _ = get_output_names(evoked)

    # Output directories
    if outdir is None:
        outdir = op.dirname(evoked)
    if not op.exists(outdir):
        os.makedirs(outdir)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.mkdir(figdir)
    
    evoked = mne.read_evokeds(evoked)
    
    if isinstance(noise_cov, str):
        noise_cov = mne.read_cov(noise_cov)
    if isinstance(fwd, str):
        fwd = mne.read_forward_solution(fwd)
    if isinstance(signal_cov, str):
        signal_cov = mne.read_cov(signal_cov)
        
    if fwd_normal:
        print('Converting forward solution to surface normals')
        fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
        
    if method == 'LCMV':
        assert signal_cov is not None, 'Signal covariance must be provided with LCMV beamformer'
        
    # Timings
    try:
        assert len(twin) is 2
        tidx = evoked[0].time_as_index(twin)
    except TypeError:
        twin = [twin, twin]
        tidx = evoked[0].time_as_index(twin)
        tidx[1] += 1
    tslice = slice(*tidx)
    tms = np.asarray(twin)*1e3
    
    print('Computing inverse solution')
    print('Method    : {}'.format(method))
    
    stcs = list()
    for evo in evoked:
        condition = evo.comment
        print('Condition : {}'.format(condition))
        
        # Plot whitening
        fig = evo.plot_white(noise_cov, show=False)
        fig.set_size_inches(8,10)
        fig.savefig(op.join(figdir, '{}_whitened').format(basename))
        plt.close('all')
        
        #evo.pick_types(meg=False, eeg=True)
        
        evoc = evo.copy()
        chs = compute_misc.pick_func_channels(evo.info)
        for ch, picks in chs.items():
            print('Inverting {} channels'.format(ch))
            evo = evo.pick_channels([evo.ch_names[i] for i in picks])
            
            if method in ['MNE', 'dSPM', 'sLORETA']:
                #
                # Minimum norm estimate
                #
                
                # Make an MEG inverse operator
                print("Making inverse operator")
                inv = mne.minimum_norm.make_inverse_operator(evo.info, fwd, noise_cov,
                                                             loose=None, depth=None,
                                                             fixed=fwd_normal)
                
                # Estimate SNR
                print("Estimating SNR at {} ms".format(tms))
                snr, snr_est = mne.minimum_norm.estimate_snr(evo, inv)
                snr_estimate = snr[tslice].mean()
                print('Estimated SNR at {} ms is {:.2f}'.format(tms, snr_estimate))
            
                # Apply the operator
                lambda2 = 1/snr_estimate**2
                
                print("Preparing inverse operator")
                inv = mne.minimum_norm.prepare_inverse_operator(inv, evo.nave, lambda2, method)
                
                print("Applying inverse operator")
                stc = mne.minimum_norm.apply_inverse(evo, inv, lambda2, method, prepared=True)
                
                print('Computing cost function')
                cost, we, jp = compute_MNE_cost(evo, inv, lambda2, twin, return_parts=True)
                print('Cost : {}'.format(cost))
                
            elif method == 'LCMV':
                stc = mne.beamformer.lcmv(evo, fwd, noise_cov, signal_cov, reg=0.05,
                                          pick_ori=None)
            
            if crop:
                print('Cropping to {}'.format(twin))
                stc.crop(*twin) # to save space
                
            print("Writing source estimate")
            stc_name = op.join(outdir, basename+'_{}'.format(ch))
            stc.save(stc_name)
            stcs.append(stc_name)
            
            
            np.savetxt(op.join(outdir, 'cost_'+basename+'_{}'.format(ch)+'.txt'), np.asarray([we, jp, cost]))
        
            evo = evoc.copy()
        # if plot:
        #     ...
        
        #
        #idx = stc.data.mean(1).argsort()[::-1][:100]
        #plt.figure()
        #plt.plot(stc.times, stc.data[idx].T)
        #plt.show()
        
    return cost
    
def prepare_simulated(sd, config, i=None):
    
    if i is None:
        i = 0
    
    d = getd(sd)
    raw = getf(d['runs'][0], '{}*_raw.fif'.format(config['prefix']))[0]
    noise_cov = getf(d['cov'], '*_raw_noise-cov.fif')[0] # general noise cov
    fwdd = d['fwds'][i]
    fwdb = op.basename(fwdd)
    fwd = getf(op.join(fwdd, 'gain'), '*-fwd.fif')[0]
    outdir = op.join(d['contrasts'], 'Simulation_{}'.format(fwdb))
    
    print('Simulating evoked data using {:s}'.format(fwdb))
    evoked, stc = simulate_evoked_data(raw, fwd, noise_cov, config['nave'],
                                  outdir=outdir)
    return evoked, stc

def simulate_evoked_data(raw, fwd, noise_cov, nave=30, outdir=None):
    """
    
    raw : instance of Raw
        Instance of raw with information corresponding to that of the forward
        solution. This is used as a template for the simulated data.
    fwd : mne.forward.Forward
        Instance of fwd used to simulate the measured data.
    noise_cov : mne.cov.Covariance
    
    nave : int
        Number of averages in evoked data. This determines the SNR as noise is
        reduced by a factor of sqrt(nave).
    
    nave = (1 / 10 ** ((actual_snr - snr)) / 20) ** 2
    
    
        
    """
    
    # Output directories
    if outdir is None:
        outdir = op.dirname(raw)
    if not op.exists(outdir):
        os.makedirs(outdir)
    figdir = op.join(outdir, 'figures')
    if not op.exists(figdir):
        os.mkdir(figdir)
        
    base = get_output_names(raw)[0]
    base += '-ave'
        
    #### raw as template
    raw = mne.io.read_raw_fif(raw)
    info = raw.info
    # 'remove' SSS projector
    info['proc_history'] = []
    info['proj_id'] = None
    info['proj_name'] = None
    
    fwd = mne.read_forward_solution(fwd, force_fixed=True, surf_ori=True)
    noise_cov = mne.read_cov(noise_cov)
    
    # Make autocorrelations in the noise using an AR model of order n
    # (get the denominator coefficients only)
    iir_filter = mne.time_frequency.fit_iir_model_raw(raw, order=5, tmin=30, tmax=30+60*4)[1]
    iir_filter[1:] /= np.ceil(np.abs(iir_filter).max()) # unstable filter..?
    #iir_filter[np.abs(iir_filter) > 1] /= np.ceil(np.abs(iir_filter[np.abs(iir_filter) > 1]))
    iir_filter=None
    
    #rng = np.random.RandomState(42)
    
    # Time axis
    start, stop = -0.2, 0.5  
    sfreq = raw.info['sfreq']
    times = np.linspace(start, stop, np.round((stop-start)*sfreq).astype(int))
    
    
    # Source time course
    np.random.seed(42)
    stc = simulate_sparse_stc(fwd['src'], n_dipoles=1, times=times,
                              random_state=42, data_fun=sim_er)
    
    # Noisy, evoked data
    
    #chs = mne.io.pick.channel_indices_by_type(info)
    
    # Pick MEG and EEG channels
    meg = [info['ch_names'][i] for i in mne.pick_types(info, meg=True, ref_meg=False)]
    eeg = [info['ch_names'][i] for i in mne.pick_types(info, meg=False, eeg=True, ref_meg=False)]
    

    # Simulate evoked data
    # simulate MEG and EEG data separately, otherwise the whitening is messed
    # up, then merge
    #noise_cov2 = noise_cov.copy()
    #noise_cov2.update(dict(data=noise_cov.data + np.random.random(noise_cov.data.shape) * noise_cov.data))
    evoked = simulate_evoked(mne.pick_channels_forward(fwd, meg), stc, info,
                             mne.pick_channels_cov(noise_cov, meg),
                             nave=nave, iir_filter=iir_filter)
    evoked_eeg = simulate_evoked(mne.pick_channels_forward(fwd, eeg), stc,
                                 info, mne.pick_channels_cov(noise_cov, eeg),
                                 nave=nave, iir_filter=iir_filter)
    evoked.add_channels([evoked_eeg])
    evoked.set_eeg_reference(projection=True)#.apply_proj()
    
    #evoked.data = mne.filter.filter_data(evoked.data, sfreq, None, 80, fir_design='firwin')
    evoked.comment = 'Simulation'
    #evoked.crop(-0.2, 0.5)
    #stc.crop(-0.2, 0.5)
    
    evoked.save(op.join(outdir, base + '.fif'))
    
    #picks = mne.pick_types(evoked.info, meg=False, eeg=True)
    fig = evoked.plot(spatial_colors=True, show=False)
    fig.savefig(op.join(figdir, base + '.png'))
    fig = evoked.plot_white(noise_cov, show=False)
    fig.savefig(op.join(figdir, base + '_whitened.png'))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(stc.times*1e3, stc.data.T*1e9)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (nAm)')
    ax.set_title('Simulated Sources')
    fig.savefig(op.join(figdir, 'simulated_sources.png'))
    
    plt.close('all')
    
    return evoked, stc

    
    

def sim_er(times):
    """Function to generate random source time courses"""
    
    sine = 50e-9 * np.sin(30. * times + np.random.randn(1)*200)
    
    peak = 0.2
    peakshift = 0.05
    duration = 0.01 # standard deviation of gaussian
    gaussian = np.exp(-(times - peak + peakshift * np.random.randn(1)) ** 2 / duration)
    return sine * gaussian
            
    
    
    
def plot_stc(stc, src, twin, vol=None):
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
    tidx = stc.time_as_index(twin)
    tms = [np.round(t*1e3).astype(int) for t in twin]
    tms = '_'.join([str(t) for t in tms])
    
    
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
            v, f = io_misc.read_surface(s)
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
            vtk = io_misc.as_vtk(v, f, **d)
            vtk_name = op.join(figdir, basename+'_{}-{}'.format(tms, n))
            vtk.tofile(vtk_name, 'binary')
        print('Done') 
        
    else:
        import nibabel as nib
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
        v, f = io_misc.read_surface(s)
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
    """    
    #print('Writing to VTK...', end=' ')
    #for v,f,d,n in zip(vertices, faces, data, names):  
    #    vtk = io_misc.as_vtk(v, f, **d)
    #    vtk_name = op.join(outdir, basename+'_{}_{}'.format(n, tms))
    #    vtk.tofile(vtk_name, 'binary')
    #print('Done')

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
        
        events, event_id = compute_misc.find_events(raw, noise_cov['event_codes'], channels['stim'])
    
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
        fig.save(op.join(save_fig, "{}_noise-cov_{}.pdf".format(raw_base, which)))
"""

#%%


def prepare_basis_functions(fwd, surface_lh, surface_rh, sigma=0.6, min_dist=3, exp_degree=8, random_seed=0,
                            write=True, symmetric_sources=False):
    """
    
    sigma : 
        the 'amount' of activity to propagate
    min_dist : 
        minimum distance between basis functions (counted in mesh edges)
    exp_degree :
        number of edge expansions to do for each basis function...
    random_seed : 
        For reproducible results.
    write : 
        Write
    symmetric_sources : bool
        
    
    """
    
    # convert to surface normal
    #fwd = mne.forward.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)
    
    v_lh, f_lh = io_misc.read_surface(surface_lh)
    v_rh, f_rh = io_misc.read_surface(surface_rh)
    
    print('Selecting vertices')
    centers_lh = select_vertices(f_lh, min_dist, random_seed)
    if symmetric_sources:
        centers_rh = match_opposite_hemisphere(centers_lh, surface_lh, surface_rh)
    else:
        centers_rh = select_vertices(f_rh, min_dist, random_seed)
    
    B_lh = make_basis_functions(surface_lh, centers_lh, sigma, exp_degree, write)
    B_rh = make_basis_functions(surface_rh, centers_rh, sigma, exp_degree, write)
    B = [B_lh, B_rh]
    
    if symmetric_sources:
        #B_bi = ss.vstack([B_lh, B_rh])
        B.append(ss.vstack([B_lh, B_rh]))
        
    # Support over vertices
    #plt.figure(); plt.hist(B_lh.sum(1),100); plt.show()
    
    # Number of basis functions overlapping with each vertice
    #plt.figure(); plt.hist(B_lh.getnnz(1),100); plt.show()
    
    D = project_forward_to_basis(fwd, B, make_bilateral=symmetric_sources)

    return D, B


def project_forward_to_basis(fwd, basis, make_bilateral=False):
    """Project forward solution from a source space onto a set of spatial basis
    functions. Implements
    
        D = AB
    
    where A [K x N] is the gain matrix, B [N x C] is the spatial basis
    functions, and D [K x C] is the gain matrix projected onto B.
    
    A may be [K x 3N] so B may need to be expanded to accomodate this.
    
    Here K sensors, N source locations, and C basis function.
    
    basis : list | scipy sparse matrix
        List of basis sets (each containing a number of basis functions).
        Should match the number of source spaces in 'fwd'.
    make_bilateral : bool
        Make a summed forward projection in addition to those contained in
        'basis'.
        
    """
    
    if isinstance(fwd, str):
        fwd = mne.read_forward_solution(fwd)
    if not isinstance(basis, list):
        basis = [basis]
    nbases = len(basis)
    nchs = len(fwd['info']['chs'])
    
    # Determine number of sources per location
    if fwd['source_ori'] is FIFF.FIFFV_MNE_FIXED_ORI:
        spl = 1
    elif fwd['source_ori'] is FIFF.FIFFV_MNE_FREE_ORI:
        spl = 3
    
    # Make the iteration items of source space lengths and basis functions
    src_sizes = [s['np'] for s in fwd['src']]
    if nbases is len(fwd['src']):
        iter_items = zip(src_sizes, basis)
    elif nbases is 1 and basis[0].shape[0]*spl == fwd['sol']['data'].shape[1]:
        iter_items = zip( [sum(src_sizes)]  , basis )
    else:
        raise ValueError('Basis functions do not seem to match forward solution.')
    
    print('Projecting forward solution')
    gains = list()
    initcols = slice(0,0)
    for s,b in iter_items:
        gain = np.zeros((nchs, b.shape[1]*spl))
        for i in range(spl):
            cols = slice(i+initcols.start, initcols.stop+s*spl, spl)
            gain[:,i::spl] = ss.csc_matrix.dot(fwd['sol']['data'][:,cols], b)
        gains.append(gain)
        initcols = slice(cols.stop, cols.stop)
    
    if make_bilateral:
        gains.append( sum(gains) )
    
    gains = np.concatenate(gains, axis=1)
    #fwd['sol']['data'] = gains
    #fwd['nsource'] = 
    
    return gains


def project_back_sourceestimate(invsol, basis):
    """Project an inverse solution from source space back to sensor space.
    
    invsol : 
        Inverse solution array of N sources by K time points.
    basis : list
        List of sparse arrays defininf the basis functions.
    scale : 
        Scaling vector describing the sum of the original forward solution per
        basis function (which was normalized to one). Now that we are
        projecting back, we want to get back to the same scaling.
    
    
    """
    
    # 4000 x 1 or 4000 x time...
    # invsol
    #
    if not isinstance(basis, list):
        basis = [basis]
    
    invsol = np.atleast_2d(invsol)
        
    print('Projecting source estimates to original sources space')
    
    # inverse solution in nano ampere?
    nA = 1#1e9
       
    # rows in inverse solution
    nsol = invsol.shape[0]
    # n basis functions
    nsrc = sum([b.shape[1] for b in basis])
    
    spl = nsol//nsrc
    ninvsol = list()
    initcols = slice(0,0)
    for b in basis:
        isol = np.zeros((b.shape[0]*spl, invsol.shape[1]))
        for i in range(spl):
            cols = slice(i+initcols.start, initcols.stop+b.shape[1]*spl, spl)
            isol[i::spl,:] = b.dot(invsol[cols]) * nA
        ninvsol.append(isol)
        initcols = slice(cols.stop, cols.stop)
    #ninvsol = np.concatenate(ninvsol,axis=0)    
    
    return ninvsol

def make_basis_functions(f, centers, sigma=0.6, degree=8, write=False):
    """
    
    """
    
    if isinstance(f, str):
        base, _ = op.splitext(f)
        v, f = io_misc.read_surface(f)
    else:
        write = False
        warn('Cannot write if surface is not a filename.')
    
    assert degree >= 1
    
    A = make_adjacency_matrix(f)
    A = A*sigma
    
    # Add diagonal
    A += ss.eye(*A.shape)
    
    # Keep only the basis functions we wish to use
    B = A[centers]
    for i in range(2,degree+1):
        B += B.dot(A)/i
    
    # Normalize basis functions
    B = B.multiply(1/B.sum(1))
    B = B.T.tocsc()
    
    # Threshold
    B = B.multiply(B>np.exp(-8))
    
    # Calculate some statistics on the support of each source
    support = B.getnnz(1)
    print('Source Support (# bases)')
    print('Min  : {:d}'.format(support.min()))
    print('Mean : {:.2f}'.format(support.mean()))
    print('Max  : {:d}'.format(support.max()))
    
    support = B.sum(1)
    print('Source Support (Weight)')
    print('Min  : {:.3f}'.format(support.min()))
    print('Mean : {:.3f}'.format(support.mean()))
    print('Max  : {:.3f}'.format(support.max()))
    
    if write:
        print('Writing to disk...', end=' ')
        
        origin = np.zeros(B.shape[0])
        origin[centers] = 1
        basis = np.asarray(B.sum(1)).squeeze()
        vtk = io_misc.as_vtk(v, f, pointdata=dict(origin=origin, basis=basis))
        
        vtk.tofile(base+'_basis', 'binary')        
        print('Done')

    return B

def match_opposite_hemisphere(targets, src, dst):
    """Find the vertices of 'dst' closest to the 'target' vertices in 'src'. 
    
    """
    if isinstance(src, str):
        sv, sf = io_misc.read_surface(src)
    if isinstance(dst, str):
        dv, df = io_misc.read_surface(dst)
        
    # Get offset from x=0 of src
    sx_offset = sv[:,0].max(0)
    sv[:,0] -= sx_offset

    # Get the vertices in src and mirror the coordinate around x=0
    stargets = sv[targets]
    stargets[:,0] = -stargets[:,0]
    
    # Get offset from x=0 of dst
    dx_offset = dv[:,0].min(0)
    dv[:,0] -= dx_offset
    
    # Find the closes match in dst
    tree = cKDTree(dv)
    d, dtargets = tree.query(np.atleast_2d(stargets))
    
    print('Targets found in destination')
    print('Distance (min)  : {:.2f}'.format(d.min()))
    print('Distance (mean) : {:.2f}'.format(d.mean()))
    print('Distance (max)  : {:.2f}'.format(d.max()))
    
    return dtargets

def make_adjacency_matrix(f):
    """Make sparse adjacency matrix for vertices with connections f.
    """    
    N = f.max()+1
    
    row_ind = np.concatenate((f[:,0],f[:,0],f[:,1],f[:,1],f[:,2],f[:,2]))
    col_ind = np.concatenate((f[:,1],f[:,2],f[:,0],f[:,2],f[:,0],f[:,1]))
    
    #row_ind = np.concatenate((f[:,0],f[:,0],f[:,1]))
    #col_ind = np.concatenate((f[:,1],f[:,2],f[:,2]))
    
    A = ss.csr_matrix((np.ones_like(row_ind), (row_ind, col_ind)), shape=(N,N))
    A[A>0] = 1
    
    return A

def select_vertices(f, min_dist=3, random_seed=None):
    """Select vertices from 'f' that are a minimum of 'min_dist' from each
    other.    
    """
    A = make_adjacency_matrix(f)
    idx = reshape_sparse_indices(A)
    
    # Ensure that (1) A includes diagonal, (2) A is binary
    if any(A.diagonal()==0):
        A += ss.eye(*A.shape)
    A = A.multiply(A>0)
    
    # 
    if random_seed is not None:
        assert isinstance(random_seed, Integral)
        np.random.seed(random_seed)
        
    # vertex enumerator
    venu = np.arange(A.shape[0])
    
    vertices_left = np.ones_like(venu, dtype=bool)
    vertices_used = np.zeros_like(venu, dtype=bool)
    
    #B = recursive_dot(A, A, recursions=min_dist-1)

    while any(vertices_left):
        i = np.random.choice(venu[vertices_left])
        vertices_used[i] = True
        
        # Find neighbors and remove remove those
        #if min_dist > 1:
        #    B = recursive_dot(A[i], A, min_dist-1)
        #else:
        #    B = A[i]
        #vertices_left[B.indices] = False
        
        
        vertices_left[recursive_index(idx, i, min_dist)] = False
    
    return np.where(vertices_used)[0]

def recursive_dot(A, B, recursions=1):
    """Recursive dot product between A and B, i.e.,
    
        A.dot(B).dot(B) ...
    
    as determined by the recursions arguments. If recursions is 1, this
    corresponds to the usual dot product.
    """
    
    assert isinstance(recursions, Integral) and recursions > 0
    
    if recursions is 1:
        return A.dot(B)
    else:
        return recursive_dot(A.dot(B), B, recursions-1)


def recursive_index(indices, start, recursions=1, collapse_levels=True):
    """Recursively index into 'indices' starting from (and including) 'start'.
        
    """
    assert recursions >= 0

    #start = [start] if not isinstance(start, list) else start
 
    levels = list()
    levels.append([start])

    i = 0
    while i < recursions:
        ith_level = set()
        for j in levels[i]:
            ith_level.update(indices[j])
        ith_level.difference_update(flatten(levels)) # remove elements from rings
        
        levels.append(list(ith_level))
        i+=1
    
    return flatten(levels) if collapse_levels else levels

def reshape_sparse_indices(A):
    """Reshape indices of sparse matrix to list."""
    return [A.indices[slice(A.indptr[i],A.indptr[i+1])].tolist() for i in range(len(A.indptr)-1)]

def flatten(l):
    """Flatten list of lists.
    """
    return [item for sublist in l for item in sublist]
