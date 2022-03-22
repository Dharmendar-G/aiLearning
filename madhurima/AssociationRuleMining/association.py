from zipfile import ZipFile
import pandas as pd
import os
#file_path = "C:\\Users\\MadhurimaPau_bhbpoof\\Downloads\\Into_to_NLP\\ailearning1\\aiLearning\\dataset\\associationRuleMining.zip"
# opening the zip file in READ mode
# with ZipFile(file_path, 'r') as zip:
#     # printing all the contents of the zip file
#     zip.printdir()
#     # extracting all the files
#     print('Extracting all the files now...')
#     zip.extractall()
#     print('Done!')
os.chdir("output")
dfs = [pd.read_csv(f, index_col=0)
        for f in os.listdir(os.getcwd()) if f.endswith('csv')]
finaldf = pd.concat(dfs, axis=0)
finaldf.reset_index(inplace=True)
finaldf.drop(['index'],axis=1,inplace=True)
print(finaldf.head())
