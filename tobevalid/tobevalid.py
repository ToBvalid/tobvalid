import fire

from tobevalid.mixture.gaussian_mixture import GaussianMixture
from tobevalid.mixture.invgamma_mixture import InverseGammaMixture
import tobevalid.stats.silverman as sv
import tobevalid.parsers.gparser as gp
import tobevalid.stats.pheight as ph
import os
import shutil
import numpy as np
from scipy.stats import skew
from scipy.stats import invgamma
from scipy.stats import kurtosis


def tobevalid(i, o=None, mode=1, t=1e-5):

    try:
        file_name = process_input(i)
        out = proccess_output(i, o, file_name)
        process_mode(mode)
        process_tolerance(t)

    except ValueError as e:
        return e

    (s, data) = gp.gemmy_parse(i)
    
    if s == 0:
        return "Resolution is 0"

    if len(data) <= 100:
        return "There is not sufficient amount of data to analyse, the results may left questions. Do not hesitate to contact ToBvalid team"    
    

    z = None

    if mode != 1:
        p_data = ph.peak_height(data, s) 

    if mode == 'auto':
        modes, kernel = sv.kde_silverman(p_data)
        mode = modes[0]

    if mode > 1:
        gauss = GaussianMixture(mode, tol = t)
        gauss.fit(p_data)
        z = gauss.Z
        gauss.savehtml(out, file_name)

    inv = InverseGammaMixture(mode, tol=t)
    inv.fit(data, z = z)
    inv.savehtml(out, file_name)
            
    statistics(data, inv)
    
def process_input(i):
    if not os.path.exists(i):
        raise  ValueError("Input path {} doesn't exist".format(i))

    if not os.path.isfile(i):
        raise ValueError("{} is not file".format(i))

                
    return  os.path.basename(os.path.splitext(i)[0])

def proccess_output(i, o, file_name):

    out = ""
    if o == None:
        out = os.getcwd()
    else:
        if not os.path.exists(o):
            raise ValueError("output path {} doesn't exist".format(o))

        if not os.path.isdir(o):
            raise ValueError( "{} is not directory".format(o))
        out = o
    
    out = out + "/" + file_name
   
    try:
        if os.path.exists(out) and os.path.isdir(out):
          shutil.rmtree(out)
        os.mkdir(out)
    except OSError:
        raise ValueError("Creation of the directory %s failed" % out)
    
    return out


def process_mode(mode):
    if mode == 'auto':
        return
    if not isinstance(mode, int): 
          raise ValueError("-mode has to be integer or 'auto'")

    if mode < 1: 
          raise ValueError("-mode has to be greater than zero or equal to 'auto'")    

def process_tolerance(t):
    if not isinstance(t, float) and not isinstance(t, int): 
          raise ValueError("-t has to be float")

    if t <=0 : 
        raise ValueError("-t has to be greater than zero")   

def statistics(B, inv):
    nB = len(B)
    MinB = np.amin(B)
    MaxB = np.amax(B)
    MeanB = np.mean(B)
    MedB = np.median(B)
    VarB = np.var(B)
    skewB = skew(B)
    kurtsB = kurtosis(B)
    firstQ, thirdQ = np.percentile(B,[25,75])
    alpha = inv.alpha
    beta = inv.betta
    b0 = inv.shift
        
    print('--------------------------------------------------')
    print('Parameters of B value distribution')
    print('Atom numbers   :', nB)
    print('Minimum B value:', MinB)
    print('Maximum B value:', MaxB)
    print('Mean           :', MeanB)
    print('Median         :', MedB)
    print('Variance       :', VarB)
    print('Skewness       :', skewB)
    print('Kurtosis       :', kurtsB)
    print('First quartile :', firstQ)
    print('Third quartile :', thirdQ)
    print('Alpha          :', alpha)
    print('Beta           :', beta)
    print('B0             :', b0)
    print('--------------------------------------------------')
    print(' ')           


if __name__ == '__main__':
    fire.Fire(tobevalid)
