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

def listofsoftwarevendors(url='https://www.cvedetails.com/vendor/firstchar-A/39/?sha=11360df8a7ad49b1310c871ce33d1158708f7eb4&trc=1934&order=1'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='listtable')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('listofsoftwarevendors.csv',index=False)

def listofproducts(url='https://www.cvedetails.com/product-list.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('td',class_='listtablecontainer')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('listofproducts.csv',index=False)

def browsecvebyyear(url='https://www.cvedetails.com/browse-by-date.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='stats')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('browsecvebyyear.csv',index=False)

def cvedetailscvssscore(url='https://www.cvedetails.com/cvss-score-charts.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0][:-1])
    df1.to_csv('cvedetailscvssscore.csv',index=False)

def cvssscoredistribution(url='https://www.cvedetails.com/cvss-score-distribution.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0][:-1])
    df1.to_csv('cvssscoredistribution.csv',index=False)

def top50bytotaldistinctvul(url='https://www.cvedetails.com/top-50-vendors.php?year=2022'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='listtable')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('top50bytotaldistinctvul.csv',index=False)

def cvssscoredfortop50(url='https://www.cvedetails.com/top-50-vendor-cvssscore-distribution.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('cvssscoredfortop50.csv',index=False)

def top50productsbydistinctvul(url='https://www.cvedetails.com/top-50-products.php?year=2022'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='listtable')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('top50productsbydistinctvul.csv',index=False)
    
def cvssscorefortop50productsbydistinctvul(url='https://www.cvedetails.com/top-50-product-cvssscore-distribution.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('cvssscorefortop50productsbydistinctvul.csv',index=False)
    
def top50versionsproductshighestnumberofsecurityvul(url='https://www.cvedetails.com/top-50-versions.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='listtable')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0])
    df1.to_csv('top50versionsproductshighestnumberofsecurityvul.csv',index=False)
    
    #******************************************************************************************************************

'''def cvedetailscvssscore(url='https://www.cvedetails.com/cvss-score-charts.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0][:-1])
    df1.to_csv('outputTable.csv',index=False)

def cvedetailscvssscore(url='https://www.cvedetails.com/cvss-score-charts.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0][:-1])
    df1.to_csv('outputTable.csv',index=False)

def cvedetailscvssscore(url='https://www.cvedetails.com/cvss-score-charts.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0][:-1])
    df1.to_csv('outputTable.csv',index=False)

def cvedetailscvssscore(url='https://www.cvedetails.com/cvss-score-charts.php'):
    page=requests.get(url)
    soup=BeautifulSoup(page.text,'html.parser')
    table=soup.find('table',class_='grid')
    df=pd.read_html(str(table))
    df1=pd.DataFrame(df[0][:-1])
    df1.to_csv('outputTable.csv',index=False)'''


Browse_vulnerabilities_By_Date = extract_data('https://www.cvedetails.com/browse-by-date.php')
# update columns names
cn = column_name_check(df = Browse_vulnerabilities_By_Date)
Browse_vulnerabilities_By_Date.rename(columns = cn,inplace=True)
Browse_vulnerabilities_By_Date.to_csv("DE_Browse_Vulnerabilities_By_Date.csv",index = False)

cvedetailscvssscore()