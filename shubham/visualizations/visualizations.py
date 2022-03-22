import pandas as pd
import glob
import os
print(os.getcwd())
path = "aiLearning\\dataset\\"
files = glob.glob(path + "/*.csv")
# df = pd.read_csv(glob.glob("1.sv"c))

print(files)