import pandas as pd
import os 
import re
import requests
from bs4 import BeautifulSoup


#print(os.getcwd())
# change the dir 
#os.chdir('SivaShankar/CVE_details')
#print(os.getcwd())

def extract_data(link):
    url = link
    #link = 'https://www.cvedetails.com/browse-by-date.php'
    r = requests.get(url)
    dfs = pd.read_html(r.text)

    for x in dfs:
        if 'Year' in x.columns:
            df1 = x
    return df1

def column_name_check(df):
    d = dict()
    for x in df.columns:
        if bool(re.search("Unnamed",x)):
            d[x] = df[x].unique()[0].replace(" ","_")
        elif bool(re.search("#",x)):
            d[x] = x.replace("#","No").replace("  "," ").replace(" ","_")
        else:
            d[x] = x
    return d

# extract data

def cvedetailscvssscore(url='https://www.cvedetails.com/cvss-score-charts.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0][:-1])
    df1.to_csv('outputTable.csv',index=False)

Browse_vulnerabilities_By_Date = extract_data('https://www.cvedetails.com/browse-by-date.php')
# update columns names
cn = column_name_check(df = Browse_vulnerabilities_By_Date)
Browse_vulnerabilities_By_Date.rename(columns = cn,inplace=True)
Browse_vulnerabilities_By_Date.to_csv("DE_Browse_Vulnerabilities_By_Date.csv",index = False)

cvedetailscvssscore()