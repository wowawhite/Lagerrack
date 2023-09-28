#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


def simpleGeneratorFun(xs, ys):
    for x, y in zip(xs, ys):  # for key, value in ... zip return iterator of tuples
        yield x
        yield y


# Driver code to check above generator function
'''for i, j in range(10, 100,1):
    yield i
    yield j
'''
ds_ = []
for it in range(1,10,1):
    it

data = [0.1,0,2.0 ,0.003]
resu = int(np.log2(len(data)))
#print(resu)
print(np.seterr())
