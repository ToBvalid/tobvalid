# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:13:37 2019

"""

import pheight as ph
import os
import itertools

root = "../data"
in_directory = "bfactor"
out_directory = "ph"

def convert_ph(root, in_dir, out_dir):
    in_directory = root + "/" + in_dir
    out_directory = root + "/" + out_dir
    files = list(itertools.chain(*[files for root, dirs, files in os.walk(in_directory) if  files]))
    [ph.peak_height_f(3, in_directory + "/" + file, out_directory + "/" + file) for file in files]
    

convert_ph(root, in_directory, out_directory)