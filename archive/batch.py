#!/bin/env python2.7
"""
@description:  Legacy method for iterating over various spring strengths
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: August 25, 2014
HISTORY:
"""
import argparse, math, os, sys
from subprocess import Popen 

# BEGIN MAIN
if __name__ == '__main__':
    forces = [0, 0.01, 0.1, 1, 1.5, 5, 10, 25, 100]
    for force in forces:
        cmd = './model.py -k %s -m 299 -r 7.3 -t 298 -acr -cvp'%(force)
        print cmd
        cmd = cmd.split()
        Popen(cmd, stdout=open('datasets/long_m299_r10_k%s'%(force), 'w'))
