"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import argparse
from tobvalid.mixture.gaussian_mixture import GaussianMixture
from tobvalid.mixture.invgamma_mixture import InverseGammaMixture
import tobvalid.stats.silverman as sv
import tobvalid.parsers.gparser as gp
import tobvalid.stats.pheight as ph
import tobvalid.local.analysis as lc
import tobvalid.stats.outliers as ot
import os
import shutil
import numpy as np
import json
import jsonschema
from jsonschema import validate




def tobvalid(i, o=None, m=1, p=None):

    mode = m
    try:
        file_name = process_input(i)
        out = proccess_output(i, o, file_name)
        parameters = process_p(p)
        local_params = get_local_params(parameters)
        ligand_params = get_ligand_params(parameters)
        gmm_params = get_gmm_params(parameters)
        igmm_params = get_igmm_params(parameters)
        resolution = get_plot_params(parameters)
        mode = process_mode(mode)

        (s, data, data_with_keys) = gp.gemmy_parse(i)

        if s > 20:
            return "Low resolution: not for automatic analysis. Not now."
            
        process_data(data)
    except ValueError as e:
        return e

    
    
    summury(i, out, mode, local_params, ligand_params, gmm_params, igmm_params, resolution)
    

  

    ot.print_outliers(out + "/Interquartile outliers.txt",
                      data, data_with_keys)

    data = ot.remove_outliers(data)


    lc.local_analysis(i, out,
                      r_main=local_params[0],
                      r_wat=local_params[1],
                      olowmin=local_params[2],
                      olowq3=local_params[3],
                      ohighmax=local_params[4],
                      ohighq1=local_params[5]
                      )
    lc.ligand_validation(i, out,
                         r_main=ligand_params[0],
                         olowmin=ligand_params[1],
                         ohighmax=ligand_params[2],
                         )

    z = None

    p_data = ph.peak_height(data, s)
    gauss = GaussianMixture(mode, tol=gmm_params[0], max_iter=gmm_params[1])
    gauss.fit(p_data)
    if gauss.n_modes > 1:
        z = gauss.Z[:, ::-1]
    gauss.savehtml(out, file_name, dpi=resolution)
    mode = gauss.n_modes

    inv = InverseGammaMixture(mode, tol=igmm_params[0], max_iter=igmm_params[1], ext = igmm_params[2])
    inv.fit(data, z=z)
    inv.savehtml(out, file_name, dpi=resolution)


    if inv.n_modes == 1:
        if (max(inv.alpha) > 10 or max(np.sqrt(inv.betta) > 30)):
            print("High values of alpha and/or beta parameters. Please consider the structure for re-refinement with consideraton of blur or other options")


def process_data(data):
    if min(data) < 0:
        raise ValueError(
            "Zero or minus values for B factors are observed. Please consider the structure model for re-refinement or contact the authors")
    if np.sum(data <= min(data)+0.0001*np.median(data)) >= 0.1*np.size(data):
        raise ValueError("Too many values close to minimum: Possible oversharpening case.")
    if np.sum(data >= max(data) - 0.0001*np.median(data)) >= 0.1*np.size(data):
        raise ValueError("Too many values close to maximum: Possible upper limit problem.")
    if np.size(data) < 200:
        raise ValueError("Too few atoms for analysis")
    udata,cnts = np.unique(data,return_counts=True) 
    nps = np.size(data)
  
    if np.size(udata) <= max(10,int(0.01*nps)):
        raise ValueError("Too many unique repeated values: B values may have not been refined properly")
    if max(cnts) >= 0.05*nps:
        raise ValueError("Some values are repeated more than 5% of times: B values may have not been refined properly")
    
def process_input(i):
    check_file(i)
    return os.path.basename(os.path.splitext(i)[0])


def proccess_output(i, o, file_name):

    out = ""
    if o == None:
        out = os.getcwd()
    else:
        if not os.path.exists(o):
            raise ValueError("output path {} doesn't exist".format(o))

        if not os.path.isdir(o):
            raise ValueError("{} is not directory".format(o))
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
        return mode
               
    if not isinstance(mode, int):
        try: 
            mode = int(mode) 
        except ValueError:
             raise ValueError("-m has to be positive integer or 'auto'")
       

    if mode < 1:
        raise ValueError("-m has to be greater than zero or equal to 'auto'")
    
    return mode

def process_p(config):

    if config == None:
        return
    check_file(config)

    try:
        with open(config, "r") as read_file:
            parameters = json.load(read_file)
    except ValueError:
        raise ValueError("Invalid json file: {}".format(config))

    try:
        validate(instance=parameters, schema=config_schema)
    except jsonschema.exceptions.ValidationError as err:
        raise ValueError(err)

    return parameters


def get_local_params(parameters):
    r_main = get_value(parameters, ["local", "r_main"], 4.2)
    r_wat = get_value(parameters, ["local", "r_water"], 3.2)
    olowmin = get_value(parameters, ["local", "olowmin"], 0.7)
    olowq3 = get_value(parameters, ["local", "olowq3"], 0.99)
    ohighmax = get_value(parameters, ["local", "ohighmax"], 1.2)
    ohighq1 = get_value(parameters, ["local", "ohighq1"], 1.01)

    if olowq3 <= olowmin:
        raise ValueError("olowq3 should be greater than olowmin")

    if ohighq1 >= ohighmax:
        raise ValueError("ohighq1 should be less than ohighmax")

    return (r_main, r_wat, olowmin, olowq3, ohighmax, ohighq1)

def get_ligand_params(parameters):
    r_main = get_value(parameters, ["ligand", "r_main"], 4.2)
    olowmin = get_value(parameters, ["ligand", "olowmin"], 0.7)
    ohighmax = get_value(parameters, ["ligand", "ohighmax"], 1.2)

    return (r_main, olowmin,  ohighmax)

def get_gmm_params(parameters):
    return (get_value(parameters, ["gmm", "tolerance"], 1e-05), get_value(parameters, ["gmm", "maxiteration"], 1000))


def get_igmm_params(parameters):
    return (get_value(parameters, ["igmm", "tolerance"], 1e-04), get_value(parameters, ["igmm", "maxiteration"], 1000),
    get_value(parameters, ["igmm", "ext"], "classic") )

def get_plot_params(parameters):
    return (get_value(parameters, ["plot", "dpi"], 150))    

def get_value(dict, path, default):
    if dict == None:
        return default

    result = dict
    for val in path:
        result = result.get(val, None)
        if result == None:
            break
    if result == None:
        return default
    return result


def check_file(i):
    if not os.path.exists(i):
        raise ValueError("Input path {} doesn't exist".format(i))

    if not os.path.isfile(i):
        raise ValueError("{} is not file".format(i))


config_schema = {
    "type": "object",
    "properties": {
            "gmm": {
                "type": "object",
                "properties": {
                    "maxiteration": {"type": "integer", "minimum": 1},
                    "tolerance": {"type": "number", "exclusiveMinimum": 0}
                }
            },
        "igmm": {
                "type": "object",
                "properties": {
                    "maxiteration": {"type": "integer", "minimum": 1},
                    "tolerance": {"type": "number", "exclusiveMinimum": 0},
                    "ext": {"type": "string", "enum": ["classic", "stochastic"]}
                }
        },
        "plot": {
                "type": "object",
                "properties": {
                    "dpi": {"type": "integer", "minimum": 72}
                }
        },
        "local": {
                "type": "object",
                "properties": {
                    "r_main": {"type": "number", "exclusiveMinimum": 0},
                    "r_water": {"type": "number", "exclusiveMinimum": 0},
                    "olowmin": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "olowq3": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "ohighmax": {"type": "number", "exclusiveMinimum": 1},
                    "ohighq1": {"type": "number", "exclusiveMinimum": 1}
                }
        },
        "ligand": {
                "type": "object",
                "properties": {
                    "r_main": {"type": "number", "exclusiveMinimum": 0},
                    "olowmin": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "ohighmax": {"type": "number", "exclusiveMinimum": 1}
                }
        }
    }
}

def summury(input, output, mode, local_params, ligand_params, gmm_params, igmm_params, resolution):
    print('''
If the results of the program are useful for you please cite:

"Local and global analysis of macromolecular Atomic Displacement Parameters". 
R.Masmaliyeva, K.Babai & G.Murshudov.
Acta Cryst. D76 (to be published).
a well as any specific reference in the program write-up.

:Reference2: For the theoretical interest to ADP distribution also please cite:
"Analysis and validation of macromolecular B values". 
Masmaliyeva, R. C. & Murshudov, G. N. (2019). 
Acta Cryst. D75, 505-518.

----------------------------------------------------------------------------------------------------

Input file: {}

Output directory: {}

Numder of modes: {}
----------------------------------------------------------------------------------------------------

Used parameters are listed below:\n'''.format(input, output, mode))

    print("Gaussian Mixture Model:")
    print("Tolerance: ", gmm_params[0])
    print("Max iteration: ", gmm_params[1])
    separator()
    print("Shifted Inverse Gamma Mixture Model:")
    print("Tolerance: ", igmm_params[0])
    print("Max iteration: ", igmm_params[1])
    print("EM extension: ", igmm_params[2])
    separator()
    print("Plotting parameters:")
    print("DPI: ", resolution)
    separator()
    print("Local ADP analysis parameters:")
    print("Radius for calculation neighbour's list(r_main): ", local_params[0])
    print("Radius for \"water coordination\" calculations(r_water): ", local_params[1])
    print("Criteria for marking atoms as light atoms: occ vs median < olowmin & occ vs third quartile < olowq3 (olowmin): {}, (olowq3): {}".format(local_params[2], local_params[3]))
    print("Criteria for marking atoms as heavy atoms: occ vs median > ohighmax & occ vs first quartile > ohighq1 (ohighmax): {}, (ohighq1): {}".format(local_params[4], local_params[5]))
    separator()
    print("Ligand validation parameters:")
    print("Radius for calculation neighbour's list(r_main): ", ligand_params[0])
    print("Criteria for marking atoms as light atoms: occ vs median < olowmin (olowmin): ", ligand_params[1])
    print("Criteria for marking atoms as heavy atoms: occ vs median > ohighmax (ohighmax): ", ligand_params[2])

def separator():
     print("----------------------------------------------------------------------------------------------------")


def main_func():
    parser = argparse.ArgumentParser(description='''"Local and global analysis of macromolecular Atomic Displacement Parameters".
R.Masmaliyeva, K.Babai & G.Murshudov.
Acta Cryst. D76 (to be published)''')

    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument("-i",  "--input", type=str, metavar="<pdb file>", help="Path to the pdb file.")

    parser.add_argument("-o", "--output", type=str, metavar='<output file directory>', default=None, help="Output directory.")
    parser.add_argument("-m", "--modes", metavar='<number of modes | auto>', default = 1, help="Number of modes. Must be positive integer ot 'auto'.")
    parser.add_argument("-p", "--params", type=str, metavar='<json parameter file>', default = None, help="Path to the json config file")

    args = parser.parse_args()

    if args.input == None:
        parser.error("the following arguments are required: -i/--input <pdb file>")

    return tobvalid(args.input, o=args.output, m=args.modes, p=args.params)
    
