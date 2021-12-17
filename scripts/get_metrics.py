# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:39:42 2021

@author: Alex
"""

from corescore import parameter_extraction
import os
import pandas as pd

RESULTS_DIR = '../Images/results/test_predictions'

df = pd.DataFrame(columns = ['filename'] + parameter_extraction.CORE_PARAMETERS)

for f in os.listdir(RESULTS_DIR):

    im = parameter_extraction.Image(os.path.join(RESULTS_DIR,f), 64, 100000000)
    params = im.parameters()
    row = pd.DataFrame(columns = ['filename'] + parameter_extraction.CORE_PARAMETERS, data=[params])
    
    df = df.append(row)
    
df.to_csv('../Images/results/test_parameters.csv')