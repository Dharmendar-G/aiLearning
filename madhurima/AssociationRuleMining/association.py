from zipfile import ZipFile
import pandas as pd
import os
file_path = "C:\\Users\\MadhurimaPau_bhbpoof\\Downloads\\Into_to_NLP\\ailearning1\\aiLearning\\dataset\\associationRuleMining.zip"
# opening the zip file in READ mode
with ZipFile(file_path, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')

