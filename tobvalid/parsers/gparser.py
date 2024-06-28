"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
import numpy as np
import gemmi
import warnings


def gemmy_resolution(file):
    if file.endswith('.cif') or file.endswith('.gz'):
        block = gemmi.cif.read(file).sole_block()
        s = block.find_value('_reflns.d_resolution_high')
    elif file.endswith('.pdb'):
        st = gemmi.read_structure(file)
        s = st.resolution
    if s == "":
        return -1
    return s


def gemmy_parse(file):
    try:
        st = gemmi.read_structure(file)
    except RuntimeError as e:
        raise ValueError(str(e))
    s = gemmy_resolution(file)

    if s == -1 or s == 0:
        warnings.warn(
            "Resolution is not available. Default 2.1 will be used..")
        s = 2.1

    B = []
    B_with_keys = {}
    chains = []
    residues = []
    atoms = []
    # read pdb file
    for chain in st[0]:
        chains.append(chain)
        for residue in chain:
            residues.append(residue)
            for atom in residue:
                occ = atom.occ
                if occ > 0:
                    B_key = [chain, residue, atom]
                    B_with_key = [B_key, atom.b_iso]
                    B_with_keys[len(B)] = B_with_key
                    B.append(atom.b_iso)
                    atoms.append(atom)
                else:
                    continue
    # The array named B gives you B factors of all atoms of the structure
    B = np.asarray(B)

    return (s, B, B_with_keys)

def chains(B_with_keys):
    l = len(B_with_keys)
    chv = []
    
    for i in range(l):
        ch = B_with_keys[i][0][0]
        ch = ch.name
        chv.append(ch)
    ch_names = np.ndarray.tolist(np.unique(chv))
    w = np.where(np.roll(chv,1)!=chv)[0]
    w = np.ndarray.tolist(w)
    w.append(l)    
    chB = {}   
    k = 0
    l2 = len(w)-1
    for i in range(l2):
        Bs = []
        ii = i+1
        start = w[i]
        stop = w[ii]
        for j in range(start, stop):
            Bs.append(B_with_keys[j][1])
        chB[k] = Bs
        k = k + 1
        
    return(ch_names, chB)
