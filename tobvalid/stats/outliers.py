from ..report import Report
from scipy.stats.mstats import mquantiles
from scipy.stats import iqr
import numpy as np
import re


def remove_outliers(data):
    qnt1 = mquantiles(data, prob = 0.25)
    qnt3 = mquantiles(data, prob = 0.75)
    k1 = 10 * iqr(data); 
    k2 = 2*iqr(data)
    if np.sum( (data >= np.min(data) + 0.001*np.median(data))) > 1:
        clean_index =  ((data <= (qnt3[0] + k1)) &  (data >= (qnt1[0] - k2)) &
                                                   (data >= np.min(data) + 0.001*np.median(data)))                                                  
    else:
        clean_index = ((data <= (qnt3[0] + k1)) &  (data >= (qnt1[0] - k2)))

    if np.sum( (data >= np.max(data) - 0.001*np.median(data))) > 1:
        clean_index =  np.logical_and(clean_index, ((data <= np.max(data) - 0.001*np.median(data))))
    
    clean_index = np.where(clean_index)[0]
    return (data[clean_index], clean_index)


def find_outliers(data):
    qnt1 = mquantiles(data, prob = 0.25)
    qnt3 = mquantiles(data, prob = 0.75)
    k1 = 5 * iqr(data); 
    k2 = 2*iqr(data)
    iqtout1 = np.asarray(np.where(data < (qnt1[0] - k2)))[0]
    iqtout3 = np.asarray(np.where(data > (qnt3[0] + k1)))[0]
    return (iqtout1, iqtout3)    


def print_outliers(B, B_with_keys):
    
    iqtout1, iqtout3 = find_outliers(B)
    
    report = Report("Outliers")

    if len(iqtout1) != 0:
        report.head('Interquartile outlier 1')
        for atom_idx in iqtout1:
            report.text(print_outlier(B_with_keys[atom_idx]))         

    if len(iqtout3) != 0:
        report.head('Interquartile outlier 3')
        for atom_idx in iqtout3:
            report.text(print_outlier(B_with_keys[atom_idx]))
    
    return report


def print_outlier(B):
    mystr = str(B[0])
    mystr=re.sub("\[<gemmi.", "", mystr)
    mystr=re.sub("with .+ res>, <gemmi\.", "", mystr)
    mystr=re.sub("with .+ atoms>, <gemmi\.", "", mystr)
    mystr=re.sub(">]", " B value: ", mystr)
    mystr = mystr + str(np.round(B[1], 3))
    return mystr                     
