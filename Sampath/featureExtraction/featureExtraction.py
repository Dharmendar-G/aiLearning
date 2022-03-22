import os
import re
import numpy as np
import pandas as pd

os.getcwd()

e1=pd.read_csv('../../dataset/1.csv',encoding='utf-8')
print(e1.head())