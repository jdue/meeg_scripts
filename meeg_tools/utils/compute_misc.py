import mne
from mne.io.constants import FIFF
import numpy as np
from sklearn.preprocessing import StandardScaler
from warnings import warn

def triangle_normal(vertices, faces):
    """Get normal vectors for each triangle in the mesh.

    PARAMETERS
    ----------
    mesh : ndarray
        Array describing the surface mesh. The dimension are:
        [# of triangles] x [vertices (of triangle)] x [coordinates (of vertices)].
    
    RETURNS
    ----------
    tnormals : ndarray
        Normal vectors of each triangle in "mesh".
    """    
    mesh = vertices[faces]
    
    tnormals = np.cross(mesh[:,1,:]-mesh[:,0,:],mesh[:,2,:]-mesh[:,0,:]).astype(np.float)
    tnormals /= np.sqrt(np.sum(tnormals**2,1))[:,np.newaxis]
    
    return tnormals
    
def vertex_normals(vertices, faces):
    """
    
    """
    face_normals = triangle_normal(vertices, faces)
    
    out = np.zeros_like(vertices)
    for i in range(len(faces)):
        out[faces[i]] += face_normals[i]
    out /= np.linalg.norm(out, ord=2, axis=1)[:, None]
        
    return out

def ica_unmix(ica, data, picks=None):
    """Project measurements to sources

        S = WX = A^(-1)X
        
    """
    # Number of principal components used for the ICA fit
    npc = ica.n_components_
     
    # Prewhiten
    if ica.noise_cov is None:
        data = data/ica.pre_whitener_
    else:
        data = ica.pre_whitener_ @ data
    
    # Center per channel (prior to PCA)
    if ica.pca_mean_ is not None:
        data -= ica.pca_mean_[:,None]
    
    # Project to PCA components and unmix
    sources = ica.unmixing_matrix_ @ ica.pca_components_[:npc] @ data
    
    return sources

def ica_mix(ica, sources):
    """Project sources to measurements
    
        X = AS
    
    """
    npc = ica.n_components_
    
    # (Re)mix sources and project back from PCA space
    data = ica.pca_components_[:npc].T @ ica.mixing_matrix_ @ sources
    
    # Undo centering
    if ica.pca_mean_ is not None:
        data += ica.pca_mean_[:,None]
    
    # Undo prewhitening to restore scaling of data
    if ica.noise_cov is None:
        data *= ica.pre_whitener_
    else:
        data = np.linalg.pinv(ica.pre_whitener_) @ data
        
    return data

def rls(signal, reference, M=3, ff=0.99, sigma=0.01, pad="signal",
        return_weights=False, inplace=False):
    """Recursive least squares.
    
    Clean "signal" based on "reference".
    
    Stationarity. RLS assumes a stationary (i.e., a stabil/consistent) relationship between the reference inputs (i.e., "reference")
    and the expression of these in the signal.
    
    
    PARAMETERS
    ----------
    signal : array_like
        Noisy N-dimensional signal vector.
    reference : array_like
        Reference signal array based on which "signal" is cleaned. Dimensions are L x N with L being the number of signals used 
        to clean the input signal.
    M : int
        Length of finite impulse response filter. This also denotes the first sample which will be filtered (default = 3).
    ff : float
        Forgetting factor (typically lambda) bounded between 0 and 1 (default = 0.99).
    sigma : float
        Factor used for initialization of the (inverse) sample covariance matrix (Ri = I/sigma).
    pad : "signal" | "zeros" | None
        How to pad signal to obtained an output vector of same length as input
        signal. If None, do not pad signal. If "zeros", pad with zeros. If
        "signal", pad with input signal. If "inplace" is true, this has no
        effect (default = "signal").
    return_weights : bool
        Return the history of the weights (default = False).
    inplace : bool
        Whether or not to modify the input array inplace (default = False).
    
    RETURNS
    ----------
    err : numpy.array
        The  signal array with the reference signal regressed out.
    Hs : numpy.array (optional)
        Array of weights.
    
    NOTES
    ----------
    He P., Wilson G., Russell C. 2004. Removal of ocular artifacts from electro-encephalogram by adaptive filtering.
    Med Biol Eng Comput. 2004 May 42(3):407-12.
    """
    assert (ff<=1) and (ff>=0), "Forgetting factor must be 0 =< ff =< 1"
    assert pad in [None, "zeros", "signal"]
    
    try:
        Nsig, Ns = signal.shape
    except ValueError:
        signal = signal[None,:]
        Nsig, Ns = signal.shape
    try:
        Nref, Nr = reference.shape
    except ValueError:
        reference = reference[None,:]
        Nref, Nr = reference.shape
    
    assert Ns == Nr, "Length of signal and reference signal(s) must be equal"
    N = Ns
    assert N > M, "Length of signal and reference signal(s) must be larger than M"
    
    # Standardize
    signal = signal.T
    signal_scaler = StandardScaler()
    signal_scaler.fit(signal)    
    signal = signal_scaler.transform(signal).T
    
    reference = reference.T  
    reference_scaler = StandardScaler()
    reference_scaler.fit(reference)
    reference = reference_scaler.transform(reference).T     

    # Initialize weights (flattened to vector)
    Wp = np.zeros( (Nsig, Nref*M) )

    # Initialize (inverse) reference covariance
    # R   : the weighted covariance matrix of the reference signals
    # Ri  : R(i)^(-1)
    # Rip : R(i-1)^(-1)
    Rip = np.eye(Nref*M,Nref*M)/sigma
    Rip = np.repeat(Rip[None,...], Nsig, axis=0)
    
    if not inplace:
        err = np.zeros( (Nsig, N-M) )
        
    if return_weights:
        Ws = np.zeros((Nsig, N-M, Nref*M))
    
    # Start from the Mth sample
    for i in np.arange(M,N):        
        # Eq. 23 : stack the reference signals column-wise
        r = reference[:,i-M:i].ravel()
        
        # Eq. 25 : calculate gain factor
        K = Rip @ r / (ff + r @ Rip @ r)[:,None]
        
        # Eq. 27 : a priori error (i.e., using the previous weights)
        alpha = signal[:,i] - Wp @ r
        
        # Eq. 26 : the correction factor is directly proportional to the gain vector, K, and the (a priori) error, alpha
        W = Wp + K*alpha[:,None]

        # Eq. 24 : update the (inverse) covariance matrix   
        Ri = (Rip - np.outer(K, r).reshape(Nsig, Nref*M, Nref*M) @ Rip ) / ff # np.outer ravels all inputs so reshape

        # A posteriori error (i.e., using the updated weights). This is the cleaned signal
        if inplace:
            signal[:,i] -= W @ r
        else:
            err[:,i-M] = signal[:,i] - W @ r
        
        if return_weights:
            # Collect weights
            Ws[:,i-M,:] = W
        
        # Prepare for next iteration
        Rip = Ri
        Wp = W
    
    # Convert signal back to original scale
    signal = signal_scaler.inverse_transform(signal.T).T
    
    if inplace:
        if return_weights:
            return signal, Ws
        else:
            return signal 
    else:
        # Convert error signal back to original scale
        err = signal_scaler.inverse_transform(err.T).T
        if pad == "signal":
            err = np.concatenate((signal[:,:M], err), axis=1)
        elif pad == "zeros":
            err = np.concatenate((np.zeros(Nsig,M), err))
        if return_weights:
            return err, Ws
        else:
            return err
        

def get_mri2head_transform(info, fids_mri, scalp=None, project_on_scalp=True, rm_points_below_nas=False):
    """Determine the transformation from MRI to head coordinates.
    
    info : Info
    
    fids_mri : dict
        Dictionary with entries 'nas', 'lpa', 'rpa' corresponding
        to the locations of the fiducials in the MRI coordinate
        system. Input is assumed in mm.
    scalp : dict (optional)
        Dictionary with entries 'v' (vertices) and 'f' (faces)
        describing the scalp mesh. This is in MRI coordinates
    project_on_scalp : bool
        Whether or not to project digitized points onto the scalp surface after affine registration.
    rm_points_below_nas : bool
        Whether or not to remove digitized points below the nasion. Useful for defaced MRIs.
          
    """
    # Get digitized points and labels (head coordinates)
    digs, _, fids_head = get_digitized_points(info)
    
    if rm_points_below_nas:
        print("Removing points below nasion")
        z_nas = fids_head["nas"][-1]
        
        # z_nas should be 0 (or very close to) as should z_lpa and z_rpa
        # to make sure lpa and rpa are kept, use z_nas-10*z_nas
        keep_idx = digs[:,-1] >= z_nas-10*z_nas
        remove_idx = np.where(~keep_idx)[0]
        keep_idx = np.where(keep_idx)[0]
        digs = digs[keep_idx]
    else:
        keep_idx = np.arange(len(digs))
        
    # Rows should match
    fids_mri = np.array([fids_mri["nas"], fids_mri["lpa"], fids_mri["rpa"]])
    fids_head = np.array([fids_head["nas"], fids_head["lpa"], fids_head["rpa"]])
    
    # TRY coregister_fiducials function instead?
    
    # Determine transformation from MRI to HEAD coordinates using fiducials
    print("Determining initial transformation")
    trans = mne.coreg.fit_matched_points(fids_mri, fids_head)
    trans = mne.transforms.Transform(fro=FIFF.FIFFV_COORD_MRI,
                                     to=FIFF.FIFFV_COORD_HEAD,
                                     trans=trans)

    
    scalp_trans = scalp.copy()
    scalp_trans["v"] = mne.transforms.apply_trans(trans, scalp["v"])
    """
    # Refine registration based on all the digitzed digs
    print("Refining transformation")
    
    trans_refine = mne.coreg.fit_point_cloud(digs, scalp_trans["v"], out="trans")
    trans_refine = np.linalg.inv(trans_refine)
    
    scalp_trans["v"] = mne.transforms.apply_trans(trans_refine, scalp_trans["v"])
    
    # The final MRI to HEAD transformation
    trans["trans"] = trans_refine @ trans["trans"]
    """
    
    # project digitized points on to scalp
    print("Projecting digitized points on scalp")
    scalp_trans = dict(rr=scalp_trans["v"], tris=scalp_trans["f"], ntri=len(scalp_trans["f"]),
                 use_tris=scalp_trans["f"], np=len(scalp_trans["v"]))
    _, _, digs_proj = mne.surface._project_onto_surface(digs, scalp_trans, project_rrs=True)

    # project EEG electrodes onto scalp surface
    eeg = mne.pick_types(info, meg=False, eeg=True)
    eeg_locs = np.asarray([ch["loc"][:3] for ch in info["chs"]])[eeg]
    _, _, eeg_locs_proj = mne.surface._project_onto_surface(eeg_locs, scalp_trans, project_rrs=True)
    
    return trans, eeg_locs_proj, digs_proj

    """
    # If necessary, update raw
    if rm_points_below_nas:
        for i in remove_idx[::-1]:
            del raw.info["dig"][i]        
    if scalp is not None and project_on_scalp:       
        for i in range(len(digs)):
            raw.info["dig"][i]["r"] = digs[i]
        for i,ch in enumerate(eeg):
            raw.info["chs"][ch]["loc"][:3] = eeg_locs[i]
    """     

def get_sensor_locs(info):
    """Return sensor locations in *head coordinates* from instance of Info.
    """
    chs = pick_func_channels(info)
    locs = dict()
    for ch, picks in chs.items():
        locs[ch] = np.asarray([ch["loc"][:3] for ch in info["chs"]])[picks]
        if ch in ['grad','mag']:
            locs[ch] = mne.transforms.apply_trans(info['dev_head_t'], locs[ch])
    return locs

def get_digitized_points(info):
    """Get digitized points from instance of Info.
    """
    labels = []
    digs = []
    for c,k,i,r in [(c, p["kind"],p["ident"],p["r"]) for c,p in enumerate(info["dig"])]:        
        if k == FIFF.FIFFV_POINT_CARDINAL:
            if i == FIFF.FIFFV_POINT_LPA:
                label = "LPA"
                lpa = c
            elif i == FIFF.FIFFV_POINT_NASION:
                label = "Nasion"
                nas = c
            elif i == FIFF.FIFFV_POINT_RPA:
                label = "RPA"
                rpa = c
            else:
                raise ValueError()
        elif k == FIFF.FIFFV_POINT_HPI:
            label = "{}{:d}".format("HPI", i)
        elif k == FIFF.FIFFV_POINT_EXTRA:
            label = "{}{:03d}".format("EXTRA", i)
        elif k == FIFF.FIFFV_POINT_EEG:
            label = "{}{:03d}".format("EEG", i)
        digs.append(r)
        labels.append(label)
    fids_head = dict(nas=digs[nas], lpa=digs[lpa], rpa=digs[rpa])
    
    return np.asarray(digs), labels, fids_head

def find_events(raw, event_codes=None, stim_channel=None, min_duration=5e-3):
    """Find events and 
    
    event_codes : dict
        keys : event ID
        values : int or list of ints which are then considered one event type
    
    """
    
    events = mne.find_events(raw, stim_channel, min_duration=min_duration)
    
    event_id = dict()
    for i,k in enumerate(event_codes.keys()):
        map_events = np.in1d(events[:,-1], event_codes[k])
        
        print("Mapping {} events with code(s) {} to {} ({})".format(map_events.sum(), event_codes[k], i, k))
        events[map_events,-1] = i
        event_id[k] = i
    
    # If any unassigned event codes, warn
    used = np.in1d(events[:,-1], list(event_id.values()))
    if any(~used):
        unassigned = np.unique(events[~used,-1])
        warn("The following stimuli codes were not assigned: {}. Disregarding them.".format(unassigned))
        events = events[used]
        
    return events, event_id

def pick_func_channels(info):
    
    # Get indices for different channel types
    meg = mne.pick_types(info, meg=True, ref_meg=False)
    grad = mne.pick_types(info, meg="grad", ref_meg=False)
    mag = mne.pick_types(info, meg="mag", ref_meg=False)
    eeg = mne.pick_types(info, meg=False, eeg=True, ref_meg=False)
    
    # The 'useful' channels
    chs = dict()
    if any(meg):
        if any(grad):
            chs["grad"] = grad
        if any(mag):
            chs["mag"] = mag
    if any(eeg):
        chs["eeg"] = eeg

    return chs

def detect_squid_jumps(raw, time=0.1, threshold=4, description = "bad_jump"):
    """
    
    """
    
    samps = int(time*raw.info["sfreq"])
    if samps % 2 == 1:
        samps += 1
    
    # Detection kernel
    kernel = np.concatenate((np.ones(1), -np.ones(1)))
    
    assert len(kernel) % 2 == 0
    hkl = int(len(kernel)/2) # half kernel length
        
    meg = mne.pick_types(raw.info, meg=True)
   # data = raw.get_data()[meg]
   # data -= data.mean(1)[:,None]
   # data /= data.std(1)[:,None]
    
    all_jumps = np.zeros(len(raw.times),dtype=bool)
    
    for x in raw.get_data()[meg]:
        
        x -= x.mean()
        x /= x.std()

        # Filter data
        #y = np.correlate(x, kernel, "valid")
        y = np.convolve(x, kernel, "valid")

        # zero pad in pre and post
        y = np.concatenate((np.zeros(hkl), y, np.zeros(hkl-1)))
        
        # Threshold the filtered signal
        thr1, thr2 = np.percentile(y, [1, 99])
        thr1 *= threshold
        thr2 *= threshold
        y = (y < thr1) | (y > thr2)
        
        # Expansion kernel
        # Convolve to cover the requested samples around the artifact
        kernel2 = np.ones(samps+1)
        assert len(kernel2) % 2 == 1
        
        y = np.convolve(y, kernel2, "valid") >= 1
        
        if any(y):
            hlk2 = int((len(kernel2)-1)/2)
            
            # zero pad pre and post
            y = np.concatenate((np.zeros(hlk2), y, np.zeros(hlk2) )).astype(bool)
            
            all_jumps = all_jumps | y
            
    # Annotate raw   
    jumps = np.where(np.concatenate(([0], np.diff(all_jumps.astype(int)))))[0]
    assert len(jumps)%2 == 0
    
    jumps = (jumps+raw.first_samp) / raw.info["sfreq"]
    for onset, offset in jumps.reshape(-1,2): 
        skip = False
        # check if segment is already included in an annotation
        for a in raw.annotations:
            if onset >= a['onset'] and onset <= a['onset']+a['duration'] or \
               offset >= a['onset'] and offset <= a['onset']+a['duration']:
               skip = True
               break
        # onset : time in seconds relative to first_samp (!)
        # duration : time in seconds
        #if raw.annotations is None:
        #    raw.annotations = mne.Annotations(onset, offset-onset, description)
        #else:
        if not skip:
            raw.annotations.append(onset, offset-onset, description)
