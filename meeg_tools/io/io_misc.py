

from itertools import islice
import mne
from mne.io.constants import FIFF
import nibabel as nib
import numpy as np
import os
import os.path as op
import pyvtk

from ..utils.compute_misc import triangle_normal

def read_surface(fname):
    """Load a surface mesh in either .off or .stl file format. Return the 
    vertices and faces of the mesh. If .stl file, assumes only one solid, i.e. 
    only one mesh per file.
    
    PARAMETERS
    ----------
    fname : str 
        Name of the file to be read (.off or .stl file).
    
    RETURNS
    ----------
    vertices : ndarray
        Triangle vertices.
    faces : ndarray
        Triangle faces (indices into "vertices").
    """
    #file_format = fname.split(".")[-1].lower()
    #file_format = op.splitext(fname)[1].lower()
    
    if fname.endswith('off'):
        with open(fname, "r") as f:
            # Read header
            hdr = f.readline().rstrip("\n").lower()
            assert hdr == "off", ".off files should start with OFF"
            while hdr.lower() == "off" or hdr[0] == "#" or hdr == "\n":
                hdr = f.readline()
            hdr = [int(i) for i in hdr.split()]
            
            # Now read the data
            vertices = np.genfromtxt(islice(f,0,hdr[0]))
            faces    = np.genfromtxt(islice(f,0,hdr[1]),
                                     usecols=(1,2,3)).astype(np.uint)
        
    elif fname.endswith('stl'):
        # test if ascii. If not, assume binary        
        with open(fname, "r") as f:
            if f.readline().split()[0] == "solid":
                is_binary = False
            else:
                is_binary = True
                
        if is_binary:
            with open(fname, "rb") as f:
                # Skip the header (80 bytes), read number of triangles (1
                # byte). The rest is the data.
                np.fromfile(f, dtype=np.uint8, count=80)         
                np.fromfile(f, dtype=np.uint32, count=1)[0]
                data = np.fromfile(f, dtype=np.uint16, count=-1)
            data = data.reshape((-1,25))[:,:24].copy().view(np.float32)
            vertices = data[:,3:].reshape(-1,3) # discard the triangle normals
            
        else:
            vertices = []
            with open(fname,"r") as f:
                for line in f:
                    line = line.lstrip().split()
                    if line[0] == "vertex":
                        vertices.append(line[1:])
            vertices = np.array(vertices, dtype=np.float)
        
        # The stl format does not contain information about the faces, hence we
        # will need to figure this out.
        faces = np.arange(len(vertices)).reshape(-1,3)

        # Remove vertice duplicates and sort rows by sum  
        # Seed for reproducibility (otherwise faces would be difference each
        # time the file is read)
        np.random.seed(0)
        sv = np.sum(vertices+vertices*(100*np.random.random(3))[np.newaxis,:],
                    axis=1)
        sv_arg = np.argsort(sv)
        sv_arg_rev = np.argsort(sv_arg) # reverse indexing for going back
        
        # Get unique rows, indices of these, and counts. Create the new indices
        # and repeat them
        u, u_idx, u_count = np.unique(sv[sv_arg],return_index=True,
                                      return_counts=True)
        repeat_idx = np.repeat(np.arange(len(u)), u_count)
        
        # Retain only unique vertices and modify faces accordingly
        vertices = vertices[sv_arg][u_idx]
        faces = repeat_idx[sv_arg_rev][faces]
            
    elif fname.endswith('gii'):
        gii = nib.load(fname)
        vertices, faces = gii.darrays[0].data, gii.darrays[1].data
    else:
        #raise IOError("Invalid file format. Only files of type .off and .stl are supported.")
        raise TypeError("Unsupported surface format '{}'".format(op.splitext(fname)[1][1:]))
    
    return vertices, faces
    
def write_surface(vertices, faces, fname, file_format="off", binary=True):
    """Save a surface mesh described by points in space (vertices) and indices
    into this array (faces) to an .off or .stl file.
    
    PARAMETERS
    ----------
    vertices : ndarray
        Array of vertices in the mesh.
    faces : ndarray, int
        Array describing the faces of each triangle in the mesh.
    fname : str
        Output filename.
    file_format : str, optional
        Output file format. Choose between "off" and "stl" (default = "off").
    binary : bool
        Only used when file_format="stl". Whether to save file as binary (or
        ascii) (default = True).

    RETURNS
    ----------
    Nothing, saves the surface mesh to disk.
    """
    nFaces = len(faces)
    file_format = file_format.lower()
    
    # if file format is specified in filename, use this
    if fname.split(".")[-1] in ["stl","off"]:
        file_format = fname.split(".")[-1].lower()
    else:
        fname = fname+"."+file_format
    
    if file_format == "off":
        nVertices = len(vertices)
        with open(fname, "w") as f:
            f.write("OFF\n")
            f.write("# (optional comments) \n\n")
        np.savetxt(fname,np.array([nVertices,nFaces,0])[np.newaxis,:],fmt="%u")
        np.savetxt(fname,vertices,fmt="%0.6f")
        np.savetxt(fname,np.concatenate((np.repeat(faces.shape[1],nFaces)[:,np.newaxis],faces),axis=1).astype(np.uint),fmt="%u")
    
    elif file_format == "stl":
        mesh = vertices[faces]
        tnormals  = triangle_normal(vertices, faces)
        data = np.concatenate((tnormals, np.reshape(mesh, [nFaces,9])),
                              axis=1).astype(np.float32)
        
        if binary:
            with open(fname, "wb") as f:
                f.write(np.zeros(80, dtype=np.uint8))
                f.write(np.uint32(nFaces))
                f.write(np.concatenate((data.astype(np.float32,order="C",copy=False).view(np.uint16),np.zeros((data.shape[0],1),dtype=np.uint16)),axis=1).reshape(-1).tobytes())
        else:
            with open(fname, "w") as f:
                f.write("solid MESH\n")
                for t in range(len(data)):
                    f.write(" facet normal {0} {1} {2}\n  outer loop\n   vertex {3} {4} {5}\n   vertex {6} {7} {8}\n   vertex {9} {10} {11}\n  endloop\n endfacet\n"\
                    .format(*data[t,:]))
                f.write("endsolid MESH\n")
    else:
        raise IOError("Invalid file format. Please choose off or stl.")

def as_vtk(digs, cells=None, pointdata=None, celldata=None):
    """Generate vtk data on an unstructured grid from digs (i.e., vertices)
    and cells (i.e., the faces). Optionally, append scalar and vector values
    for vertices and faces. If cells is None, a vtk object with only the digs
    will be created (this may still contained pointdata).
    
    The returned vtk object can be saved using
    
        vtk.tofile(filename, format="ascii")
        vtk.tofile(filename, format="binary")
        
    and visualized in, for example, ParaView.
    
    PARAMETERS
    ----------
    digs : ndarray | list of lists or tuples
        The vertices making up the mesh.
    cells : ndarray
        The cells generating the elements (triangles or tetrahedra) of
        the mesh.
    pointdata : dict
        Data associated with digs. Specify as field name (key) and data
        (values).
    celldata : dict
        Data associated with cells (e.g., triangles). Specify as pointdata.
        
    RETURNS
    ----------
    vtk : VtkData
        vtk data object.
    """
    
    # Ensure list of lists
    if isinstance(digs, np.ndarray):
        digs = digs.tolist()
    if isinstance(cells, np.ndarray):
        cells = cells.tolist()
      
    # Set cell type 
    if cells is None:
        # Make a cell for points such that VTK filters will work properly
        cells = dict(poly_vertex=np.arange(len(digs)))
        # cells = dict() # For compatibility below
        celldata = None
    else:
        if len(cells[0]) is 3:
            cells = dict(triangle=cells)
        elif len(cells[0]) is 4:
            cells = dict(tetra=cells)
    
    structure = pyvtk.UnstructuredGrid(digs, **cells)
    header = 'VTK data'
    vtk = pyvtk.VtkData(structure, header)
                      
    # Determine field types and assemble DataSets
    if pointdata is not None:
        assert isinstance(pointdata, dict)
        pft = dict() # field types
        for k,v in pointdata.items():
            try:
                _, d = v.shape
            except ValueError: # only one dimension
                d = 1
            except AttributeError:
                d = len(v[0]) # assume all entries are equal
            
            if d is 1:
                pft[k] = "scalar"
            elif d is 3:
                pft[k] = "vector"
            else:
                raise ValueError("must be 1 or 3...")
        
        # Assemble
        #pd = pyvtk.PointData()
        for k,v in pointdata.items():
            if pft[k] == "scalar":
                 x = pyvtk.Scalars(v,name=k)
            elif pft[k] == "vector":
                 x = pyvtk.Vectors(v,name=k)
            #pd.append(x)
            vtk.point_data.append(x)
    #else:
    #    pd = None
                
    if celldata is not None:
        assert isinstance(celldata, dict)
        cft = dict()
        for k,v in celldata.items():
            try:
                _, d = v.shape
            except ValueError: # only one dimension
                d = 1
            except AttributeError:
                d = len(v[0]) # assume all entries are equal
            
            if d is 1:
                cft[k] = "scalar"
            elif d is 3:
                cft[k] = "vector"
            else:
                raise ValueError("must be 1 or 3...")
    
        #cd = pyvtk.CellData()       
        for k,v in celldata.items():
            if cft[k] == "scalar":
                 x = pyvtk.Scalars(v,name=k)
            elif cft[k] == "vector":
                 x = pyvtk.Vectors(v,name=k)
            #cd.append(x)
            vtk.cell_data.append(x)
    #else:
    #    cd = None
    
    return vtk
    
def read_data(fname, verbose=False):
    """Read data. Supported formats are listed here
    
        http://martinos.org/mne/dev/manual/io.html
    
    PARAMETERS
    ----------
    fname : str
        Data file or directory to read.
    verbose : bool, None
        Verbosity    
    RETURNS
    ----------
    """
    
    if os.path.isdir(fname):
        try:
            data = mne.io.read_raw_ctf(fname, verbose=verbose)
        except ValueError:
            data = mne.io.read_raw_bti(fname, verbose=verbose)
        else:
            raise IOError("Unknown data format for input of type directory.")
        return data
    elif os.path.isfile(fname):
        _, ext = os.path.splitext(fname)
        ext = ext.lstrip(".")
        if ext == "fif":
            data = mne.io.read_raw_fif(fname, verbose=verbose)
        elif ext == "sqd":
            data = mne.io.read_raw_kit(fname, verbose=verbose)
        elif ext == "vhdr":
            data = mne.io.read_raw_brainvision(fname, verbose=verbose)
        elif ext == "cnt":
            data = mne.io.read_raw_cnt(fname, verbose=verbose)
        elif ext == "edf":
            data = mne.io.read_raw_edf(fname, verbose=verbose)
        elif ext == "bdf":
            data = mne.io.read_raw_bdf(fname, verbose=verbose)
        elif ext == "egi":
            data = mne.io.read_raw_egi(fname, verbose=verbose)
        elif ext == "set":
            data = mne.io.read_raw_set(fname, verbose=verbose)
        else:
            raise IOError("Unknown data format '{}'.".format(ext))
        return data
    else:
        raise IOError("Input is neither file nor directory.")

def prepare_sourcespace(pos, tris=None, coord_frame='mri', surf_id=None):
    """Setup the a discrete MNE source space object (as this is more flexible
    than the surface source space).
    
    pos :
        Source positions
    tris :
        If source positions are vertices of a surface, this defines the
        surface.
    coord_frame :
        mri or head
            
    sid : 
        surface id. lh, rh, or None.
    """
    
    # mm -> m (input assumed to be in mm)
    pos *= 1e-3
    
    # Source normals
    if tris is None:
        # Define an arbitrary direction
        nn = np.zeros((npos, 3))
        nn[:, 2] = 1.0
    else:
        nn = None # Calculate later
    
    if coord_frame == 'mri':
        coord_frame = FIFF.FIFFV_COORD_MRI
    elif coord_frame == 'head':
        coord_frame = FIFF.FIFFV_COORD_HEAD
    else:
        raise ValueError('coord_frame must be mri or head')
    
    assert surf_id in ('lh', 'rh', None)
    if surf_id == 'lh':
        surf_id = FIFF.FIFFV_MNE_SURF_LEFT_HEMI
    elif surf_id == 'rh':
        surf_id = FIFF.FIFFV_MNE_SURF_RIGHT_HEMI
    elif surf_id is None:
        surf_id = FIFF.FIFFV_MNE_SURF_UNKNOWN
        
    # Assumed to be in mm, thus mm -> m
    #pos = dict(
    #    rr = pos * 1e-3,
    #    nn = source_normals * 1e-3
    #    )
    #src = mne.setup_volume_source_space(subject=None, pos=pos, verbose=False)
    npos = len(pos)
    
    src = dict(
        id = surf_id,
        type = 'discrete',
        np = npos,
        ntri = 0,
        coord_frame = coord_frame,
        rr = pos,
        nn = nn,
        tris = None,
        nuse = npos,
        inuse = np.ones(npos),
        vertno = np.arange(npos),
        nuse_tri = 0,
        use_tris = None    
        )
    
    # Unused stuff
    src.update(dict(
        nearest = None,
        nearest_dist = None,
        pinfo = None,
        patch_inds = None,
        dist = None,
        dist_limit = None,
        subject_his_id = None
        ))
    
    if tris is not None:
        # Setup as surface source space
        # MNE doesn't like surface source spaces that are not LH or RH
        assert src['id'] in (FIFF.FIFFV_MNE_SURF_LEFT_HEMI,
                             FIFF.FIFFV_MNE_SURF_RIGHT_HEMI)
        surf = dict(rr=pos, tris=tris)
        surf = mne.surface.complete_surface_info(surf)
        
        src['type'] = 'surf'
        src['tris'] = surf['tris']
        src['ntri'] = surf['ntri']
        src['nn'] = surf['nn'] # vertex normals
        
        # we use all tris, so the following are not really used
        src['use_tris'] = None # else [nuse_tri x 3] of indices into src['tris'] 
        src['nuse_tri'] = 0    # else len(src['use_tris'])
    
    src = [src]
    
    return mne.source_space.SourceSpaces(src) 

def sourcespace_from_files(files, coord_frame='mri', surf_id=None):
    """
    If surf_id is None and the filenames starts with 'lh' or 'rh' assume left
    and right hemisphere, respectively, else set 'unknown' as source space id.
    Else surf_id show be a list of entries corresponding to the input files.
    """
    # Read and prepare the source space
    if isinstance(files, str):
        files = [files]
    assert isinstance(files, list)
    
    if surf_id is not None:
        assert len(surf_id) == len(files)
    else:
        surf_id = [None]*len(files)
    
    src = []    
    for f,this_id in zip(files, surf_id):
        # Source space id
        if this_id is None:
            base = op.basename(f)
            if base.startswith('lh'):
                this_id = 'lh'
            elif base.startswith('rh'):
                this_id = 'rh'
        
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            # Assume a (binary) nifti image
            img = nib.load(f)
            x,y,z = np.where(img.get_data() > 0)
            pos = np.concatenate((x[:,None], y[:,None], z[:,None]),axis=1)
            pos = mne.transforms.apply_trans(img.affine, coo)
            tris = None
        else:
            # Assume it is a file defining a mesh
            pos, tris = read_surface(f)
        src.append(prepare_sourcespace(pos, tris, coord_frame, this_id))
    
    if isinstance(src, list):
        src = concat_sourcespaces(src)
    
    return src

def concat_sourcespaces(src):
    """Concatenate instances of SourceSpaces.
    """
    # list is the base class of SourceSpaces so isinstance will not work
    if not isinstance(src, list):
       src = [src]
    if len(src) == 1:
        return src[0]
    return src[0]+concat_sourcespaces(src[1:])

    
def read_pos(filename):
    """Read coordinates from a Polhemus device.
    """
    with open(filename, "r") as f:
        ndig = np.int(f.readline())
        digs = []
        for i in range(ndig):
            digs.append(np.asarray(f.readline().split()[1:], dtype=np.float64))
            
        nasion = f.readline().split()
        lpa = f.readline().split()
        rpa = f.readline().split()
    
    # Digitized points
    digs = np.asarray(digs)
    #digs = digs[:,[1,0,2]]

    # Convert from cm to m
    digs *= 1e-2
    
    assert nasion[0] == "nasion"
    assert lpa[0] == "left"
    assert rpa[0] == "right"
    
    nasion = np.asarray(nasion[1:], dtype=np.float64)*1e-2
    lpa = np.asarray(lpa[1:], dtype=np.float64)*1e-2
    rpa = np.asarray(rpa[1:], dtype=np.float64)*1e-2
    
    # stack
    #hpi = np.concatenate((nasion[None,:], lpa[None,:], rpa[None,:]), axis=0)#[:,[1,0,2]]
    
    
    #hpi *= 1e-2
    
    return nasion, lpa, rpa, digs
