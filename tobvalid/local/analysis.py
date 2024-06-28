"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"


This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import gemmi
import numpy as np
from scipy.special import erf
import pandas as pd
import warnings
from ..report import Report

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def occupancy_estimate(b1, b2, smax):
    ccm = (b2/b1)**1.5
    if smax > 0 and b1 > 0 and b2 > 0:
        cc1 = np.sqrt(np.pi)*erf(np.sqrt(b1)*smax/2) - \
            np.sqrt(b1)*smax*np.exp(-b1*smax**2/4)
        cc2 = np.sqrt(np.pi)*erf(np.sqrt(b2)*smax/2) - \
            np.sqrt(b2)*smax*np.exp(-b2*smax**2/4)
        ccm = ccm*cc1/cc2
    return(ccm)


def atna(Bval, Bn, smax, ml):
    bmedian = np.median(Bn)
    bsq = 0
    for bn in Bn:
        bsq = bsq + bn**2
    bsq = bsq/ml
    bq1, bq3 = np.percentile(Bn, [25, 75])
    ccm = 1
    ccq1 = 1
    ccq3 = 1
    if bmedian > 0:
        b01 = bmedian + Bval
        b02 = 2*bmedian
        ccm = occupancy_estimate(b01, b02, smax)

        iqr = bq3-bq1
        bb1 = max(0.001, bmedian-2*iqr)
        b01 = bb1+Bval
        b02 = 2*bb1
        ccq1 = occupancy_estimate(b01, b02, smax)

        bb3 = bmedian+2*iqr
        b01 = bb3+Bval
        b02 = 2*bb3
        ccq3 = occupancy_estimate(b01, b02, smax)

    return(ccm, ccq1, ccq3, bq1, bq3)


def parse_an(myprot, r_main):
    st = gemmi.read_structure(myprot)
    s = st.resolution
    name = st.name
    #r_main = 4.2
    if s == "" or s == 0:
        warnings.warn(
            "Resolution is not available. Default 2.1 A will be used..")
        s = 2.1
    B = []
    B_with_keys = {}
    chains = []
    residues = []
    # atoms = []
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
                    # atoms.append(atom)
                else:
                    continue
    B = np.asarray(B)
    subcells = gemmi.NeighborSearch(st[0], st.cell, r_main)
    # subcells.populate()
    for n_ch, chain in enumerate(st[0]):
        for n_res, res in enumerate(chain):
            for n_atom, atom in enumerate(res):
                subcells.add_atom(atom, n_ch, n_res, n_atom)
    return(st[0], s, B_with_keys, B, subcells, residues, name)


def el2list(cras, las):
    mybool1 = False
    for ls in las:
        if ls == cras:
            mybool1 = True
            break
    return(mybool1)


def l2df(mylist, mydf):
    mybool = False
    l = len(mydf.columns)
    for i in range(l):
        xx = mylist == mydf[i]
        u = xx.unique()
        mybool = bool(u.sum) == True and u.size == 1
        if mybool == True:
            break
    return(mybool)


def ligand_validation(input_path, r_main=4.2, olowmin = 0.7, ohighmax = 1.2):
    
    st, s, B_with_keys, B, subcells, residues, name = parse_an(input_path, r_main)
    smax = 1/s
    ligands = []
    ligand_report = Report("Analysis results for all ligands within the structure model")
    ligand_report.head("")
    ligand_report.text("The list of neighbours are calculated using the radius {r}A".format(r = r_main))
    
    ligand_validation_report = Report("List of ligans with unusual character")
    ligand_validation_report.head("")
    ligand_validation_report.text("The list of neighbours are calculated using the radius {r}A".format(r = r_main))


    
    for r in residues:
        if r.het_flag == 'H' and r.name != 'HOH':
            ligands.append(r)

    for l in ligands:
        tc1 = str(l.subchain)[0:1]
        slas = []
        B_las = []
        rej = pd.DataFrame()
        for la in l:
            ms = str(la.serial.imag)
            ta = str(la.name)
            slas.append(la.serial)
            B_las.append(la.b_iso)
        ccm1 = np.median(B_las)
        Bn = []
        for la in l:
            ta = str(la.name)
            tri = str(l.seqid)
            tc = str(l.subchain[0:1])
            ref_atom = st[tc][tri][0][ta][0]
            marks = subcells.find_neighbors(
                ref_atom, min_dist=0.1, max_dist=4.2)
            for mark in marks:
                cras = []
                cra = mark.to_cra(st)
                ms = str(mark.image_idx)
                scras = cra.atom.serial
                mybool1 = el2list(scras, slas)
                if mybool1 == False:
                    ta = str(cra.atom.name)
                    tri = str(cra.residue.seqid)
                    tr = str(cra.residue.name)
                    tc = str(cra.chain.name)
                    cras = [tc, tri, tr, ta, ms]
                    mybool = l2df(cras, rej)
                    if mybool == False:
                        Bn.append(cra.atom.b_iso)
                        rej[len(rej.columns)] = cras

        ccm2 = np.median(Bn)
        ccm_ph = occupancy_estimate(ccm1, ccm2, smax)
        ccm_td = occupancy_estimate((ccm1+ccm2), (ccm2*2), smax)

        ligand_report.text("pdb: {pdb} , chain: {chain} , residue: {name} , serial: {n} ,  occ = {occ} , PH = {ph}, ligand B: {ccm1} , neighs B: {ccm2}".format(
            pdb=name, chain=tc1, name=l.name, n=l.seqid, occ=round(ccm_td, 2), ph=round(ccm_ph, 2), ccm1=round(ccm1, 2), ccm2=round(ccm2, 2)))

        if ccm_td < olowmin or ccm_td > ohighmax:

            ligand_validation_report.text("pdb: {pdb} , chain: {chain} , residue: {name} , serial: {n} ,  occ = {occ} , PH = {ph}, ligand B: {ccm1} , neighs B: {ccm2}".format(
            pdb=name, chain=tc1, name=l.name, n=l.seqid, occ=round(ccm_td, 2), ph=round(ccm_ph, 2), ccm1=round(ccm1, 2), ccm2=round(ccm2, 2)))
    return [ligand_report, ligand_validation_report]

def local_analysis(input_path, r_main = 4.2, r_wat = 3.2, olowmin = 0.7, olowq3 = 0.99, ohighmax = 1.2, ohighq1 = 1.01):
    


    st, s, B_with_keys, B, subcells, residues, mypdb = parse_an(input_path, r_main)
    smax = 1/s
    nB = len(B)
    res_lh = {}
    water = {}

    #ligands = []
    
    local_report = Report("Local B value analysis within the radius of {r}A \n\n".format(r = r_main))
    local_report.head("")


    water_report = Report("Water molecules with number of neighbours 6 and more")
    water_report.head("")
    water_report.texts("Water molecules with number of neighbours 6 and more are listed below:")
    water_report.texts("The list of neighbours are calculated using the radius {r}A".format(r = r_wat))


    phl = ["Potentially lighter atom", "Potentially heavier atom"]
    j = 0
    w = 0
    
    for i in range(nB):
        tc = B_with_keys[i][0][0].name
        tri = str(B_with_keys[i][0][1].seqid)
        tr = B_with_keys[i][0][1].name
        ta = B_with_keys[i][0][2].name
        # ta1 = B_with_keys[i][0][2]
        # ref_atom = st.sole_residue(tc, gemmi.SeqId(tri))[ta]
        ref_atom = st[tc][tri][0][ta][0]
        marks = subcells.find_neighbors(
            ref_atom, min_dist=0.1, max_dist=r_main)
        marks1 = subcells.find_neighbors(
            ref_atom, min_dist=0.1, max_dist=r_wat)
        Bval = B_with_keys[i][1]
        ml = len(marks)
        ml1 = len(marks1)
        Bn = []
        cras = []
        if ml >= 3:
            for mark in marks:
                cra = mark.to_cra(st)
                Bn.append(cra.atom.b_iso)
                cras.append(cra.atom.serial)

            bav = np.mean(Bn)
            bst = np.std(Bn)
            if bst > 0:
                bz = (Bval-bav)/bst
            bmedian = np.median(Bn)

            ccm, ccq1, ccq3, bq1, bq3 = atna(Bval, Bn, smax, ml)

            if ccq1 > ohighq1 and ccm > ohighmax or ccm < olowmin and ccq3 < olowq3:
                if ccq3 < 0.99 and ccm < 0.8:
                    my_phl = 0
                if ccq1 > 1.01 and ccm > 1.2:
                    my_phl = 1
                Bns = [[None for y in range(ml)] for x in range(3)]
                for h in range(ml):
                    Bns[0][h] = marks[h].to_cra(st).atom.serial
                    Bns[1][h] = marks[h].to_cra(st).atom.name
                    Bns[2][h] = round(marks[h].to_cra(st).atom.b_iso, 2)
                res_lh[j] = {
                    'local': {'hl': my_phl, 'occ': round(ccm, 2)},
                    'atom': {'atom': ta, 'residue': tr, 'chain': tc, 'residue_number': tri, 'B_factor': round(Bval, 2)},
                    'stats': {'neighbour_num': ml, 'meanB': round(bav, 2), 'std': round(bst, 2), 'z_value': round(bz, 2), 'B_median': round(bmedian, 2), '1stQ': round(bq1, 2), '3rdQ': round(bq3, 2)},
                    'neighbours': Bns
                }

                j = j + 1

        if tr == 'HOH':
            if ml1 >= 6:
                Bn1 = []
                for mark1 in marks1:
                    cra1 = mark1.to_cra(st)
                    Bn1.append(cra1.atom.b_iso)
                bst1 = np.std(Bn1)
                if bst1>0:
                    #bz = (Bval-bav)/bst
                    bmedian1 = np.median(Bn1)
                
                ccm, ccq1, ccq3, bq1, bq3 = atna(Bval, Bn1, smax, ml1)
                water[w] = {'atom': ta, 'residue number': tri, 'residue': tr, 'chain': tc, 'B factor': round(Bval, 2), 'neighbours number': ml1, 'neighbours median B': round(bmedian1, 2), 'occ' :round(ccm, 2)}

    for i in res_lh:

        
        local_report.texts("=============================================================================")
        local_report.texts("{phl} with optimal occupancy :  {occ}".format(
            phl=phl[res_lh[i]['local']['hl']], occ=res_lh[i]['local']['occ']))
        local_report.texts("     atom: {atom} residue: {rn} {r} chain: {chain}, B value: {B}".format(
            atom=res_lh[i]['atom']['atom'], rn=res_lh[i]['atom']['residue_number'], r=res_lh[i]['atom']['residue'], chain=res_lh[i]['atom']['chain'], B=res_lh[i]['atom']['B_factor']), 1)
        local_report.texts("Neighbors:", 1)
        
       
        for k in range(len(res_lh[i]['neighbours'][0])):
             local_report.texts("atom: {serial}  {atom}, B value: {B}".format(
                serial=res_lh[i]['neighbours'][0][k], atom=res_lh[i]['neighbours'][1][k], B=res_lh[i]['neighbours'][2][k]), 2)
        
        local_report.texts("")
        local_report.texts("Basic statistics:", 1)
        local_report.texts("number of neighbors: {an_num}".format(an_num=res_lh[i]['stats']['neighbour_num']), 2)
        local_report.texts("mean B value: {meanB}".format(meanB=res_lh[i]['stats']['meanB']), 2)
        local_report.texts("STD: {std}".format(std=res_lh[i]['stats']['std']), 2)
        local_report.texts("z value: {z}".format(z=res_lh[i]['stats']['z_value']), 2)
        local_report.texts("median: {median}".format(median=res_lh[i]['stats']['B_median']), 2)
        local_report.texts("1st quartile: {fstQ}".format(fstQ=res_lh[i]['stats']['1stQ']), 2)
        local_report.texts("3rd quartile: {trdQ}".format(trdQ=res_lh[i]['stats']['3rdQ']), 2)
        

    for w in water:
        water_report.texts(str(water[w]))
    

    return [local_report, water_report]
