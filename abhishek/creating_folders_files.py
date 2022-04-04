import numpy as np
import os
import pandas as pd

folder_list = ['aashish', 'basha', 'gopal','madhurima', 'mukund', 'prakash', 'sheshivardhan', 'shivasaikumar', 'shweta', 'srikar',
'abhishek', 'dharmendra', 'lakshmideepthi', 'manasa', 'pavan', 'sampathvinay', 'shilpa', 'shubham', 'sivashankar']

def create_folder_file(folder_list):
    source_dir = os.getcwd()
    for item in folder_list:
        if os.path.exists(os.path.join(source_dir, item)) == False:
            os.mkdir(item)
            p = os.path.join(source_dir, item)
            with open(os.path.join(p, "test.py"),'w'):
                pass
        else:
            p = os.path.join(source_dir, item)
            with open(os.path.join(p, "test.py"),'w'):
                pass
        #print('folder already there')