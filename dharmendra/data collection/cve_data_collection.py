import pandas as pd
import requests
from bs4 import BeautifulSoup

url = 'https://www.cvedetails.com/'

page = requests.get(url)
tables = pd.read_html(page.content)
c1, c2 = 'Number Of Vulnerabilities','CVSS Score'
table = [t for t in tables if c1 and c2 in t][0]
data = pd.DataFrame(table[:10])

data.to_csv('cve_data.csv', index=False)

print(data)