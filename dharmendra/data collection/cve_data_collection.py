import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

# website
url = 'https://www.cvedetails.com/'
page = requests.get(url)
table = pd.read_html(page.content)[6]
data = pd.DataFrame(table[:10])
for c in data.columns:
    try:
        data[c] = data[c].astype('float')
    except Exception as e:
        continue
print(data)
data.to_csv('cve_data.csv', index=False)

# Visualization for vulnerabilities distribution 
ax = data.plot(kind='bar', figsize=(10,6), 
               title='Vulnerability Distribution by CVSS Scores', 
               ylabel=data.columns[1], legend=False)
ax.margins(y=0.1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.bar_label(ax.containers[0])
plt.show();
