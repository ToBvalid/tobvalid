#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:35:32 2019

@author: bioprogrammer
"""
import numpy as np 
import gemmi 

myprot = "/path/to/the/file/xxxx.pdb"
st = gemmi.read_structure(myprot)
s=st.resolution
if s == "":
    print("Resolution is not available. Default 2A will be used..")
    s = 2
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
B = np.asarray(B)