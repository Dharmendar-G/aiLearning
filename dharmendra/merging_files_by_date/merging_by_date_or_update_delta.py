# Merging files by given date range or adding new delta to the existing final data
# To Run this unzipped files from AssociationRuleMining.zip is required 
# but that is causing merge conflicts due to 25k files, so I have given my localpath which wont work here 
import os,time,datetime
import zipfile
import pandas as pd

# function for Unzipping files
def unzip(file):
    with zipfile.ZipFile(os.getcwd()+f'\\{file}', 'r') as zip_obj:
        zip_obj.extractall(os.getcwd())

## Method 1 
# Unzipping files and merging with list comprehensions
start = time.time()
unzip('associationRuleMining.zip')
dfs = [pd.read_csv(f"{os.getcwd()}\\output\\{f}", index_col=0) for f in os.listdir(os.getcwd()+"\\output") if f.endswith('csv')]
finaldf = pd.concat(dfs, axis=0)
finaldf.reset_index(inplace=True)
finaldf.drop(['index'],axis=1,inplace=True)
finaldf.to_csv('finaldf.csv', index=False)
end = time.time()
# execution time in seconds 
print(f"Total number of csv files  :  {len(dfs)}\n")
print(f"time taken to merge all the csv's into single dataframe: {end-start} sec")

## Method 2
# without unzipping directly merging files list comprehensions 
start = time.time()
path = f"{os.getcwd()}\\associationRuleMining.zip"
zf = zipfile.ZipFile(path)
with zf as thezip:
    a = thezip.infolist() # list of filenames
    b = [pd.read_csv(thezip.open(a[x].filename,mode='r')) for x in range(1,len(a))]
    finaldf = pd.concat(b, axis=0)
end = time.time()
# execution time in seconds 
print(f"Total number of csv files  :  {len(a)}\n")
print(f"Time taken to merge all the csv's into single dataframe: {end-start} sec")

# Function for Merging files based on datetime ranges
path = 'C:\\Users\\DharmendraGa_5wskc\\AI Training BS\\Tasks\\csv merging'

# Merging files based on datetime ranges and updating the new delta 
def merge_files(path, from_date, to_date, base=None, update_previous=None):
    start = time.time()
    files = []
    if base=='modified':
        for file in os.listdir('output'):
            m_time = os.path.getmtime(f"{path}\\output\\{file}")
            dt_m = datetime.datetime.fromtimestamp(m_time)
            if (dt_m>=datetime.datetime.fromisoformat(from_date)) and (dt_m<=datetime.datetime.fromisoformat(to_date)):
                files.append(file)
            else:
                continue
    elif base=='created':
        for file in os.listdir('output'):
            c_time = os.path.getctime(f"{path}\\output\\{file}")
            dt_c = datetime.datetime.fromtimestamp(c_time)
            if (dt_c>=datetime.datetime.fromisoformat(from_date)) and (dt_c<=datetime.datetime.fromisoformat(to_date)):
                files.append(f"{os.getcwd()}\\output\\{file}")
            else:
                continue
    try:
        dfs = [pd.read_csv(f) for f in files]
        finaldf = pd.concat(dfs, axis=0)
        finaldf.drop('Unnamed: 0', axis=1, inplace=True)
        
        if update_previous==True:
            prev_finaldf = pd.read_csv('finaldf.csv')
            update_finaldf = pd.concat([finaldf, prev_finaldf], axis=0)
            update_finaldf.to_csv('finaldf.csv', index=False)
            end = time.time()
            print(f"\nTotal '{len(files)}' csv files created from {from_date} to {to_date}")
            print(f"\nTime taken to update the new csv's into final dataframe: {end-start} seconds")
            return update_finaldf
        end = time.time()
        print(f"\nTotal '{len(files)}' csv files created from {from_date} to {to_date}")
        print(f"\nTime taken to merge the csv's into single dataframe: {end-start} seconds")
        return finaldf
    except:
        return None

merge_files(path, "2022-03-23 14:32:08.858096", "2022-03-23 14:32:08.862004", base='created', update_previous=True)

