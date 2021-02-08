#!/bin/env python2.7
"""
A handy functio to read in something and calculate it's hydrodynamic radius
"""
import math, os, sys
import MDAnalysis
import numpy as np

u = MDAnalysis.Universe('cod.pdb')
N = len(u.atoms)
rij = MDAnalysis.core.distances.self_distance_array(u.atoms.get_positions())

R = np.sum(1/rij)/N**2
print 1/R
