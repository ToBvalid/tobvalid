from scipy.stats.mstats import mquantiles
from scipy.stats import iqr
import numpy as np
import re

def remove_outliers(data):
    qnt1 = mquantiles(data, prob = 0.25)
    qnt3 = mquantiles(data, prob = 0.75)
    k = 3 * iqr(data)
    clean_data = data[(data <= (qnt3[0] + k)) &  (data >= (qnt1[0] - k))]
    return clean_data


def find_outliers(data):
    qnt1 = mquantiles(data, prob = 0.25)
    qnt3 = mquantiles(data, prob = 0.75)
    k = 3 * iqr(data)
    iqtout1 = np.asarray(np.where(data < (qnt1[0] - k)))[0]
    iqtout3 = np.asarray(np.where(data > (qnt3[0] + k)))[0]
    return (iqtout1, iqtout3)    


def print_outliers(filename, B, B_with_keys):
    iqtout1, iqtout3 = find_outliers(B)
    
    outF = open(filename, "w")
    
    if len(iqtout1) != 0:
        outF.write('----------Interquartile outlier 1----------------')
        outF.write("\n")
        for atom_idx in iqtout1:
            print_outlier(outF, B_with_keys[atom_idx])
        outF.write("\n")            

    if len(iqtout3) != 0:
        outF.write('----------Interquartile outlier 3----------------')
        outF.write("\n")
        for atom_idx in iqtout3:
            print_outlier(outF, B_with_keys[atom_idx])
        outF.write("\n")
    outF.close()    


def print_outlier(outF, B):
    mystr = str(B)
    mystr=re.sub("\[\[<gemmi.", "", mystr)
    mystr=re.sub("with .+ res>, <gemmi\.", "", mystr)
    mystr=re.sub("with .+ atoms>, <gemmi\.", "", mystr)
    mystr=re.sub(">],", " B value:", mystr)
    mystr=re.sub("]", "", mystr)
    outF.write(mystr)
    outF.write("\n")                      