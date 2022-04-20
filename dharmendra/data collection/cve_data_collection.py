import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pymysql
import mysql.connector
from sqlalchemy import create_engine
import sqlalchemy

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

# Database details
host_name = 'localhost'
user_name = 'root'
password = '011012#SQL'
database = 'CVE_DATA'
table_name = 'cve_details'

# For Inserting data into database 
def insert_to_database(data):
    # Connecting to database
    mydb = mysql.connector.connect(host=host_name, user=user_name, password=password)
    mycursor = mydb.cursor()
    # Creating Database if not exists 
    mycursor.execute(f"CREATE database IF NOT EXISTS {database}")
    # Inserting into MySQL Database
    db_engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://{user_name}:{password}@{host_name}/{database}')
    data.to_sql(con=db_engine, name=table_name, if_exists='replace')
    print(f"Data Successfully Inserted into '{database}' Database!")

# Getting data from database
def get_database_table(table_name):
    db_engine = sqlalchemy.create_engine(f'mysql+mysqlconnector://{user_name}:{password}@{host_name}/{database}')
    database_table = pd.read_sql_table(table_name, db_engine)
    return database_table


# website
url = 'https://www.cvedetails.com/'

# Getting Data
df = get_data(url)

# Visualizing
visualize_data(df)

# Inserting into database 
insert_to_database(df)

# Getting Data from database
get_database_table(table_name)

