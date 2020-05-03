#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:31:55 2019
version: 0.0.4
@author: bioprogrammer
"""
import sys
import gemmi
import numpy as np
from scipy.special import erf
import os
import warnings
#import scipy as sp

def atan(Bval, Bn, smax, ml):
    bmedian = np.median(Bn)
    bsq = 0
    for bn in Bn:
        bsq = bsq + bn**2
    bsq = bsq/ml
    bq1, bq3 = np.percentile(Bn,[25,75])
    ccm = 1
    ccq1 = 1
    ccq3 = 1
    if bmedian>0:
        b01 = bmedian + Bval
        b02 = 2*bmedian
        ccm = (b02/b01)**1.5    
        if smax > 0:
            cc1 = np.sqrt(np.pi)*erf(np.sqrt(b01)*smax/2) - np.sqrt(b01)*smax*np.exp(-b01*smax**2/4)
            cc2 = np.sqrt(np.pi)*erf(np.sqrt(b02)*smax/2) - np.sqrt(b02)*smax*np.exp(-b02*smax**2/4)
            ccm = ccm*cc1/cc2
        iqr = bq3-bq1
        bb1 = max(0.001, bmedian-2*iqr)
        b01 = bb1+Bval
        b02 = 2*bb1
        ccq1 = (b02/b01)**1.5
        if smax>0:
            cc1 = np.sqrt(np.pi)*erf(np.sqrt(b01)*smax/2) - np.sqrt(b01)*smax*np.exp(-b01*smax**2/4)
            cc2 = np.sqrt(np.pi)*erf(np.sqrt(b02)*smax/2) - np.sqrt(b02)*smax*np.exp(-b02*smax**2/4)
            ccq1 = ccq1*cc1/cc2
            #if cc2==0:
                #print("Attention!!! b02 is zero",b02)
        bb3 = bmedian+2*iqr
        b01 = bb3+Bval
        b02 = 2*bb3
        ccq3 = (b02/b01)**1.5
        if smax>0:
            cc1 = np.sqrt(np.pi)*erf(np.sqrt(b01)*smax/2) - np.sqrt(b01)*smax*np.exp(-b01*smax**2/4)
            cc2 = np.sqrt(np.pi)*erf(np.sqrt(b02)*smax/2) - np.sqrt(b02)*smax*np.exp(-b02*smax**2/4)
            ccq3 = ccq3*cc1/cc2
    return(ccm, ccq1, ccq3, bq1, bq3)
    
def local_analysis(input_path, ouput_path):

    st = gemmi.read_structure(input_path)
    s=st.resolution

    unus_path = ouput_path + "/"+st.name+"_unus.txt"
    water_path =  ouput_path + "/"+st.name+"_water.txt"
    if s == "" or s == 0:
        warnings.warn("Resolution is not available. Default 2A will be used..")
        s = 2
    if s > 0:
        smax=1/s
    B = []
    B_with_keys={}
    chains = []
    residues = []
    atoms = []  
    heavy = []
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
    nB = len(B)
    subcells = gemmi.SubCells(st[0], st.cell, 4.2)
    for n_ch, chain in enumerate(st[0]):
            for n_res, res in enumerate(chain):
                for n_atom, atom in enumerate(res):
                    subcells.add_atom(atom, n_ch, n_res, n_atom)
    #,,,
    # if smax>0 
    #...
    for i in range(nB):
        tc = B_with_keys[i][0][0].name
        tri = str(B_with_keys[i][0][1].seqid)
        #tr = B_with_keys[i][0][1].name
        ta = B_with_keys[i][0][2].name
        #ref_atom = st[0].sole_residue(tc, gemmi.SeqId(tri))[ta]
        ref_atom = st[0].sole_residue(tc, gemmi.SeqId(tri)).sole_atom(ta)
        marks = subcells.find_neighbors(ref_atom, min_dist=0.1, max_dist=4.2)
        Bval = B_with_keys[i][1]
        ml = len(marks)
        Bn = []
        cras = []
        if ml >= 2:
            for mark in marks:
                cra = mark.to_cra(st[0])
                Bn.append(cra.atom.b_iso)
                cras.append(cra.atom.serial)
            ccm, ccq1, ccq3, bq1, bq3 = atan(Bval, Bn, smax, ml)
            if ccq1>1.01 and ccm>1.2:
                heavy.append(ref_atom.serial)
                
    F = open(unus_path, "w")
    F.close()
    F2 = open(water_path, "w")
    F2.close() #check in the end, if it is empty delete the file    XXX
        
    for i in range(nB):
        tc = B_with_keys[i][0][0].name
        tri = str(B_with_keys[i][0][1].seqid)
        tr = B_with_keys[i][0][1].name
        ta = B_with_keys[i][0][2].name
        #ref_atom = st[0].sole_residue(tc, gemmi.SeqId(tri))[ta]
        ref_atom = st[0].sole_residue(tc, gemmi.SeqId(tri)).sole_atom(ta)
        marks = subcells.find_neighbors(ref_atom, min_dist=0.1, max_dist=4.2)
        Bval = B_with_keys[i][1]
        ml = len(marks)
        Bn = []
        cras = []
        if ml >= 3:
            for mark in marks:
                cra = mark.to_cra(st[0])
                Bn.append(cra.atom.b_iso)
                cras.append(cra.atom.serial)
            outns = np.array(list(set(cras).intersection(set(heavy))))
            loutns = len(outns)
            
            for j in outns:
                xx = cras.index(j)
                #print(xx)
                #Bn.remove(Bn[xx])
                Bn[xx] = -1
            h = 0
            while h < len(Bn):
                if Bn[h] == -1:
                    Bn.remove(Bn[h])
                h = h + 1
            tempav = np.mean(Bn)
            if loutns > 0:
                for i in range(loutns):
                    Bn.append(tempav)
                
            bav = np.mean(Bn)
            bst = np.std(Bn)
            if bst>0:
                bz = (Bval-bav)/bst
            bmedian = np.median(Bn)
            
            ccm, ccq1, ccq3, bq1, bq3 = atan(Bval, Bn, smax, ml)
            
            if ccq1>1.01 and ccm>1.2:
                F = open(unus_path, "a")
                F.write(">> Potentially heavier atom with occupancy = "+str(round(float(ccm), 2))+"\n summ   "+str(ml)+"  "+str(round(float(bav), 2))+"  "+str(round(float(bst), 2))+"  "+str(round(float(bz), 2))+"  "+str(round(float(bmedian), 2))+"   "+str(round(float(bq1), 2))+"  "+str(round(float(bq3), 2))+"  "+str(round(float(ccm), 2))+"  "+str(round(float(ccq1), 2))+"  "+str(round(float(ccq3), 2))+"\natom "+str(i+1)+" "+ta+" . "+tr+" "+tri+" . "+tc+"    "+str(round(Bval, 2))+"    "+str(round(s, 2))+"\n")
                F.close()
                Bns = [ [ None for y in range( ml ) ] for x in range( 3 ) ]
                for h in range(ml):
                    Bns[0][h] = marks[h].to_cra(st[0]).atom.serial
                    Bns[1][h] = marks[h].to_cra(st[0]).atom.name
                    Bns[2][h] = round(marks[h].to_cra(st[0]).atom.b_iso, 2)
                    F = open(unus_path, "a")
                    F.write("\n.atom "+str(Bns[0][h])+".:"+str(Bns[1][h])+":."+str(Bns[2][h]))
                    F.close()
                F = open(unus_path, "a")
                F.write("\n------------------------------------------------------------\n\n")
                F.close()
            if ccq3<0.99 and ccm<0.8:
                F = open(unus_path, "a")
                F.write(">> Potentially lighter atom with occupancy = "+str(round(float(ccm), 2))+"\n summ   "+str(ml)+"  "+str(round(float(bav), 2))+"  "+str(round(float(bst), 2))+"  "+str(round(float(bz), 2))+"  "+str(round(float(bmedian), 2))+"   "+str(round(float(bq1), 2))+"  "+str(round(float(bq3), 2))+"  "+str(round(float(ccm), 2))+"  "+str(round(float(ccq1), 2))+"  "+str(round(float(ccq3), 2))+"\natom "+str(i+1)+" "+ta+" . "+tr+" "+tri+" . "+tc+"    "+str(round(Bval, 2))+"    "+str(round(s, 2))+"\n")
                F.close()
                Bns = [ [ None for y in range( ml ) ] for x in range( 3 ) ]
                for h in range(ml):
                    Bns[0][h] = marks[h].to_cra(st[0]).atom.serial
                    Bns[1][h] = marks[h].to_cra(st[0]).atom.name
                    Bns[2][h] = round(marks[h].to_cra(st[0]).atom.b_iso, 2)
                    F = open(unus_path, "a")
                    F.write("\n.atom "+str(Bns[0][h])+".:"+str(Bns[1][h])+":."+str(Bns[2][h]))
                    F.close()
                    F = open(unus_path, "a")
                    F.write("\n------------------------------------------------------------\n\n")
                    F.close()

            if tr == 'HOH':
                if ml >= 8:
                    F2 = open(water_path, "a")
                    F2.write("atom "+str(i+1)+"  "+ta+"  "+tr+"  "+tri+"  "+tc+"  "+str(round(Bval, 2))+"  .:. "+str(ml)+"\n")
                    F2.close()
        
    fbul1 = os.stat(unus_path).st_size == 0
    if fbul1 == True:
        os.remove(unus_path)
    fbul2 = os.stat(water_path).st_size == 0
    if fbul2 == True:
        os.remove(water_path)
    if fbul1 == True and fbul2 == True:
        warnings.warn("This model has no unusual atoms")
   

