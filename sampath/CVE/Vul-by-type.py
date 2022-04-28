#importing libraries
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import csv

#data collection function
def newFunction(pageno,year):
    url='https://www.cvedetails.com/vulnerability-list.php?vendor_id=0&product_id=0&version_id=0&page={0}&hasexp=0&opdos=0&opec=0&opov=0&opcsrf=0&opgpriv=0&opsqli=0&opxss=0&opdirt=0&opmemc=0&ophttprs=0&opbyp=0&opfileinc=0&opginf=0&cvssscoremin=0&cvssscoremax=0&year={1}&month=0&cweid=0&order=1&trc=7777&sha=45d566efbc1f55ce107b057217e11d794a7bc4fb'.format(pageno,year)
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='searchresults sortable')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('VulByType-{0}-{1}.csv'.format(pageno,year),index=False)

#post processing
year=2022
for pageno in range(1,157):
    newFunction(pageno,year)
    df1=pd.read_csv('VulByType-{0}-{1}.csv'.format(pageno,year))
    df1.drop(['Unnamed: 15','Unnamed: 16','Unnamed: 17','Unnamed: 18','Unnamed: 19'],axis=1,inplace=True)
    des1=df1.loc[1::2,'CVE ID'].values
    df1=df1[0::2]
    df1.reset_index(inplace=True)
    df1.drop(['#','index'],axis=1,inplace=True)
    df1['Description']=des1
    df1.to_csv('VulByType-{0}-{1}.csv'.format(pageno,year))

#combined_csv 
filenames=[i for i in os.listdir() if i.startswith('VulByType')]
combined_csv = pd.concat( [ pd.read_csv(f,index_col=[0]) for f in filenames ], axis=0)
combined_csv.to_csv('VulByType-{0}-{1}.csv'.format('all',year))