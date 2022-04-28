import requests
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2


# Get data from website url

def get_data(url):
    page = requests.get(url)
    table = pd.read_html(page.content)[6]
    data = pd.DataFrame(table[:10])
    for col in data.columns:
        try:
            data[col] = data[col].astype('float')
        except Exception as ex:
            continue
    return data

# For Visualization the data
def visualize_data(data):
    ax = data.plot(kind='bar', figsize=(10,6),
                   title='Vulnerability Distribution by CVSS Scores',
                   ylabel=data.columns[1], legend=False)
    ax.margins(y=0.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.bar_label(ax.containers[0])
    plt.show()


# website
url = 'https://www.cvedetails.com/'

# Getting Data
df = get_data(url)

# Visualizing
visualize_data(df)

# Saving to local storage
df.to_csv('cve_data.csv', index=False)


conn = psycopg2.connect(database="postgres",
                        user='postgres', password='Sreemontini@2013',
                        host='127.0.0.1', port='5432')
# create a cursor object
# cursor object is used to interact with the database
cur = conn.cursor()

# open the csv file using python standard file I/O
# copy file into the table just created
with open('cve_data.csv', 'r') as f:
    next(f)                               # Skip the header row.
    cur.copy_from(f, table='cve_details', sep=',')
    # Commit Changes
    conn.commit()
    # Close connection
    conn.close()


f.close()