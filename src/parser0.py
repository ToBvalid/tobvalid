#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:12:20 2019

@author: bioprogrammer
"""

def parse0(file):
    F = open(file, "r")
    b = F.read().split('\n')
    F.close()
    l = len(b)
    s = b[0]
    B = []
    for x in range(1, l):
        B.append(float(b[x]))
    return(s, B)
