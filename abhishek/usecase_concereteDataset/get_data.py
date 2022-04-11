import os
import pandas as pd

def get_Data():
    path = os.getcwd()
    files = []
    for file in os.listdir():
        if file.endswith(".csv"):
            files.append(open(file))
    data = pd.read_csv(files[0])  
    return data
    