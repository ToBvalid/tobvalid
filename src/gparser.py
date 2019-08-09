#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:35:32 2019

@author: bioprogrammer
"""

"""
install gemmi library
https://gemmi.readthedocs.io/en/latest/install.html
note: please use github clone rather than pip installation
"""
import numpy as np 
import gemmi 

def gemmy_resolution(file):
    st = gemmi.read_structure(file)
    s=st.resolution
    if s == "":
        return -1
    return s

def gemmy_parse(file):
    st = gemmi.read_structure(file)
    B = []
    B_with_keys={}
    chains = []
    residues = []
    atoms = []
    #read pdb file    
    for chain in st[0]:
        chains.append(chain)
        for residue in chain:
            residues.append(residue)
            for atom in residue:
                occ = atom.occ
                if occ > 0:
                    B_key  = [chain, residue, atom]
                    B_with_key = [B_key, atom.b_iso]
                    B_with_keys[len(B)]=B_with_key
                    B.append(atom.b_iso)
                    atoms.append(atom)
                else: continue 
    # The array named B gives you B factors of all atoms of the structure
    B = np.asarray(B)
    s=st.resolution
    return (s, B)
    
 
#myprot = "../data/pdb/1BDQ_out.pdb"
#st = gemmi.read_structure(myprot)
## s  IS RESOLUTION
#s=st.resolution
#if s == "":
#    print("Resolution is not available. Default 2A will be used..")
#    s = 2
#B = []
#B_with_keys={}
#chains = []
#residues = []
#atoms = []
##read pdb file    
#for chain in st[0]:
#    chains.append(chain)
#    for residue in chain:
#        residues.append(residue)
#        for atom in residue:
#            occ = atom.occ
#            if occ > 0:
#                B_key  = [chain, residue, atom]
#                B_with_key = [B_key, atom.b_iso]
#                B_with_keys[len(B)]=B_with_key
#                B.append(atom.b_iso)
#                atoms.append(atom)
#            else: continue 
## The array named B gives you B factors of all atoms of the structure
#B = np.asarray(B) 
