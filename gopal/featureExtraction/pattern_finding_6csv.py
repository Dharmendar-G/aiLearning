import pandas as pd
import re
import os

print(os.getcwd())

df=pd.read_csv("../../mukund/StandardTemplate.csv", encoding='cp1252')
df2=pd.read_csv("../../dataset/6.csv")

b = []
for i in df2["new_col"].astype(str):
    b.append(i)

w=list(set(b))

def find_phone_number(text):
    ph_no = re.findall(r"[\+\(]?[8-9][0-9 .\-\(\)]{8,}[0-9]",str(text))
    return "".join(ph_no)

def find_producer_nu(text):
    nu=re.findall(r'(Producer Number _)\s+([0-9]+)',str(text))
    return nu[0][1]

def policy_num(text):
    pol_num=re.findall(r'([pP]olicy number _)\s+([\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9])',str(text))
    return pol_num[0][1]

def city(text):
    city=re.findall(r'(City/State/Zipcode:)\s+([a-zA-Z]+\s+[a-zA-Z,]+\s([a-zA-Z]+))',str(text))
    return city[0][1].split(",")[0]

csv6=[]
csv6.append(find_producer_nu(w))
csv6.append(0)
csv6.append(find_phone_number(w))
csv6.append(0)
csv6.append(0)
csv6.append(policy_num(w))
csv6.append(0)
csv6.append(city(w))
csv6.append(0)

df3 = pd.read_csv("../../dataset/5.csv")

c = []
for i in df3["new_col"].astype(str):
    c.append(i)
e = list(set(c))

def city1(text):
    city=re.findall(r'(City / State / Zipcode)\s+([a-zA-Z]+\s+[a-zA-Z,]+\s([a-zA-Z]+\s([a-zA-z]+))+)',str(text))
    return city[0][1].split(",")

def policy_num(text):
    pol_num=re.findall(r'(policy number |)\s+([\+\(]?[1-2][0-9 .\-\(\)]{8,}[0-9])',str(text))
    return pol_num[0][1]

csv5=[]
csv5.append(0)
csv5.append(0)
csv5.append(0)
csv5.append(0)
csv5.append(0)
csv5.append(policy_num(e))
csv5.append(0)
csv5.append(city1(e)[0])
csv5.append(0)

df4 = pd.read_csv("../../dataset/4.csv")

z = []
for i in df4["new_col"].astype(str):
    z.append(i)
r = list(set(z))

def find_producer_nu1(text):
    nu=re.findall(r'(Producer Number)\s+([0-9]+)',str(text))
    return nu[0][1]

def city1(text):
    city=re.findall(r'(City/State/Zipcode)\s+([a-zA-Z]+\S+\s+[a-zA-Z]+)',str(text))
    return city[0][1]

def city(text):
    city=re.findall(r'(City/State/Zipcode)\s+([a-zA-Z]+\s+[a-zA-Z,]+\s([a-zA-Z]+))',str(text))
    return city[0][1]

def find_phone_number1(text):
    ph_no = re.findall(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]",str(text))
    return ph_no[0]

def policy_num2(text):
    pol_num=re.findall(r'(policy number _)+([\+\(]?[0-9][0-9 .\-\(\)]{8,}[0-9])',str(text))
    return pol_num[0][1]

csv4=[]
csv4.append(find_producer_nu1(r))
csv4.append(0)
csv4.append(find_phone_number1(r))
csv4.append(0)
csv4.append(0)
csv4.append(policy_num2(r))
csv4.append(0)
csv4.append(city1(r).split()[0])
csv4.append(0)

df5=pd.read_csv("../../dataset/3.csv")

def fax_num(text):
    fax=re.findall(r'(Fax:)\s+([\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9])',str(text))
    return fax


x = []
for i in df5["new_col"].astype(str):
    x.append(i)
t = list(set(x))

def find_producer_nu3(text):
    nu=re.findall(r'(PRODUCER NO:)\s+([0-9]+)',str(text))
    return nu[0][1]


def contact_info(text):
    contact = re.findall(r'(CONTACT INFO:)\s+([a-zA-Z].+).+(Phone)', str(text))
    return contact[0][1]

def phone_num3(text):
    phn_num=re.findall(r'([\+\(]?[8-9][0-9 .\-\(\)]{8,}[0-9])',str(text))
    return phn_num

def city3(text):
    city=re.findall(r'(City:)\s+([a-zA-Z]+\s+[a-zA-Z]+)',str(text))
    return city[0][1]

def country_req(text):
    country=re.findall(r'(County: (requirep) __)\s+([a-zA-z].+)',str(text))
    return country

def street(text):
    street=re.findall(r'(Street:)\s+([0-9a-zA-Z].+)',str(text))
    return street[0][1].split(",")[0]

def policy_num3(text):
    pol_num=re.findall(r'(POLICY NO.)\s+([\+\(]?[0-9][0-9 .\-\(\)]{6,}[0-9])',str(text))
    return pol_num[0][1]

def insured_name(text):
    r3=re.findall(r'(NAME: ___)+([a-zA-Z].+(CHUBB POLICY))',str(text))
    return r3[0][1]

csv3=[]
csv3.append(find_producer_nu3(t))
csv3.append(contact_info(t))
csv3.append(phone_num3(t)[0])
csv3.append(fax_num(t)[0][1])
csv3.append(insured_name(t))
csv3.append(policy_num3(t))
csv3.append(street(t))
csv3.append(city3(t))
csv3.append(0)

df6 = pd.read_csv("../../dataset/2.csv")

m = []
for i in df6["new_col"].astype(str):
    m.append(i)
y = list(set(m))

def insured_name(text):
    name=re.findall(r'(Insured Name:)\s+([a-zA-Z]+\s+[a-zA-Z]+).+(Chubb)',str(text))
    return name[0][1]

def policy_num4(text):
    policy=re.findall(r'(Policy No.:)\s+([0-9][a-zA-Z]+)',str(text))
    return policy[0][1]

def street(text):
    street=re.findall(r'(Street:)\s+([0-9a-zA-Z].+)',str(text))
    return street[0][1].split(",")[0]

def country_req(text):
    country=re.findall(r'(Counntty [(]required[)])\S([a-zA-z].+)',str(text))
    return country[0][1].split(":")[0]

csv2=[]
csv2.append(0)
csv2.append(0)
csv2.append(phone_num3(y)[0])
csv2.append(fax_num(y)[0][1])
csv2.append(insured_name(y))
csv2.append(policy_num4(y))
csv2.append(street(y))
csv2.append(city3(y))
csv2.append(country_req(y))

df7 = pd.read_csv("../../dataset/1.csv")

n = []
for i in df7["new_col"].astype(str):
    n.append(i)
u = list(set(n))

def country_req1(text):
    country=re.findall(r'(County [(]required[)]\S+\s([a-zA-Z]+))',str(text))
    return country[0][1]

def city3(text):
    city=re.findall(r'(City:)\s+([a-zA-Z]+)',str(text))
    return city[0][1]

csv1=[]
csv1.append(0)
csv1.append(0)
csv1.append(phone_num3(u)[0])
csv1.append(fax_num(u)[0][1])
csv1.append(insured_name(u))
csv1.append(policy_num(u))
csv1.append(street(u))
csv1.append(city3(u))
csv1.append(country_req1(u))

lst=['PRODUCER NO',"CONTACT INFO","Phone.1","Fax.1","INSURED’S NAME","CHUBB POLICY NO.","Street.1","City.1","County: (REQUIRED)"]

for i, j in enumerate(lst):
    df.loc[0, j] = csv1[i]

for i, j in enumerate(lst):
    df.loc[1, j] = csv2[i]

for i, j in enumerate(lst):
    df.loc[2, j] = csv3[i]

for i, j in enumerate(lst):
    df.loc[3, j] = csv4[i]

for i, j in enumerate(lst):
    df.loc[4, j] = csv5[i]

for i, j in enumerate(lst):
    df.loc[5, j] = csv6[i]


df.to_csv("submission_StandardTemplate.csv")


