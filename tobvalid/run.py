"""
Author: "Rafiga Masmaliyeva, Kaveh Babai, Garib N. Murshudov"

    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

import fire


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
        process_mode(mode)

        (s, data, data_with_keys) = gp.gemmy_parse(i)
        process_data(data)
    except ValueError as e:
        return e


    if len(data) <= 100:
        return "There is not sufficient amount of data to analyse, the results may left questions. Do not hesitate to contact ToBvalid team"

    ot.print_outliers(out + "/Interquartile outliers.txt",
                      data, data_with_keys)

    data = ot.remove_outliers(data)

    if len(data) <= 100:
        return "There is not sufficient amount of data to analyse, the results may left questions. Do not hesitate to contact ToBvalid team"


    lc.local_analysis(i, out,
        r_main = local_params[0],
        r_wat = local_params[1],
        olowmin = local_params[2],
        olowq3 = local_params[3],
        ohighmax = local_params[4],
        ohighq1 = local_params[5]
        )
    lc.ligand_validation(i, out,
        r_main = get_value(parameters, ["ligand", "r_main"], 4.2),
        olowmin = get_value(parameters, ["ligand", "olowmin"], 0.7),
        ohighmax = get_value(parameters, ["ligand", "ohighmax"], 1.2),
    )

    z = None

    p_data = ph.peak_height(data, s)
    gauss = GaussianMixture(mode, 
        tol=get_value(parameters, ["gmm", "tolerance"], 1e-05), 
        max_iter=get_value(parameters, ["gmm", "maxiteration"], 200))
    gauss.fit(p_data)
    if gauss.n_modes > 1:
        z = gauss.Z[:, ::-1]
    gauss.savehtml(out, file_name, dpi=get_value(parameters, ["plot", "resolution"], 150))
    mode = gauss.n_modes

    inv = InverseGammaMixture(mode, 
        tol=get_value(parameters, ["igmm", "tolerance"], 1e-05), 
        max_iter=get_value(parameters, ["igmm", "maxiteration"], 200))
    inv.fit(data, z=z)
    inv.savehtml(out, file_name, dpi=get_value(parameters, ["plot", "resolution"], 150))

    if inv.n_modes == 1:
        if (max(inv.alpha) > 10 or max(np.sqrt(inv.betta) > 30)):
            print("High values of alpha and/or beta parameters. Please consider the structure for re-refinement with consideraton of blur or other options")


def process_data(data):
    if min(data) < 0:
        raise ValueError(
            "Zero or minus values for B factors are observed. Please consider the structure model for re-refinement or contact the authors")


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
        return
    if not isinstance(mode, int):
        raise ValueError("-m has to be integer or 'auto'")

    if mode < 1:
        raise ValueError("-m has to be greater than zero or equal to 'auto'")


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
    r_wat = get_value(parameters, ["local", "r_wat"], 3.2)
    olowmin = get_value(parameters, ["local", "olowmin"], 0.7)
    olowq3 = get_value(parameters, ["local", "olowq3"], 0.99)
    ohighmax = get_value(parameters, ["local", "ohighmax"], 1.2)
    ohighq1 = get_value(parameters, ["local", "ohighq1"], 1.01)

    if olowq3 <= olowmin :
        raise ValueError("olowq3 should be greater than olowmin")

    if ohighq1 >= ohighmax :
        raise ValueError("ohighq1 should be less than ohighmax")

    return (r_main, r_wat, olowmin, olowq3, ohighmax, ohighq1)

def get_value(dict, path, default):
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
                    "tolerance": {"type": "number", "exclusiveMinimum": 0}
                }
            },
            "plot": {
                "type": "object",
                "properties": {
                    "resolution": {"type": "integer", "minimum": 72}
                }
            },
            "local": {
                "type": "object",
                "properties": {
                    "r_main": {"type": "number", "exclusiveMinimum": 0},
                    "r_wat": {"type": "number", "exclusiveMinimum": 0},
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

def main_func():
    fire.Fire(tobvalid)


if __name__ == '__main__':
    fire.Fire(tobvalid)
