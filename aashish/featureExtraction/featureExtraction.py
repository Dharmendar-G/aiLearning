import pandas as pd
import re
import nltk
nltk.download('punkt')

main=pd.read_csv(r"StandardTemplate.csv",encoding="cp1252")

def agency_name_helper(st):
  #print(st)
  strl=st.lower()
  id=strl.find('agency name')
  i=strl.index(':')
  las=strl.find("Chubb Agency Code".lower())
  #print(id,i,las)
  #print(st)
  if las==-1:
    return st[i+1:]
  else:
    return st[i+1:las]
def agency_name(par):
    ls=list(set(par))
    ls=list(filter(lambda x:type(x)==str,ls))
    for i,j in enumerate(ls):
      if 'agency name' in j.lower():
        #print(j)
        return(agency_name_helper(j))


def agency_code_helper(st):
    id = st.find('Legacy Chubb Agency Code'.lower())
    i = st.rfind(':')
    return st[i + 1:]


def agency_code(par):
    ls = list(set(par))
    for i, j in enumerate(ls):
        try:
            if 'Legacy Chubb Agency Code'.lower() in j.lower():
                return (agency_code_helper(j.lower()))
                # print(j)
        except:
            # print(j)
            pass


def Chubb_Policy_No_helper(st):
    id = st.find('Chubb Ltd Policy No'.lower())
    i = st.rfind(':')
    return st[i + 1:]


def Chubb_Policy_No(par):
    ls = list(set(par))
    for i, j in enumerate(ls):
        try:
            if 'Chubb Ltd Policy No'.lower() in j.lower():
                return (Chubb_Policy_No_helper(j))
                # print(j)
        except:
            # print(j)
            pass


def street_helper(st):
    id = st.find('street:'.lower())
    i = st.rfind(':')
    return st[id + 8:]


def street(par):
    ls = list(set(par))
    for i, j in enumerate(ls):
        try:
            if 'street:'.lower() in j.lower():
                # print(j)
                return (street_helper(j))
        except:
            # print(j)
            pass


def fax_helper(st):
    id = (st.lower()).find('fax:'.lower())
    # print(id)
    i = st.rfind(':')
    las = (st.lower()).find('phone'.lower())
    # print(las)
    if las == -1:
        return st[i + 1:]
    elif las != -1 and id < las:
        return st[id + 4:las]
    else:
        return st[id + 4:]


def fax(par):
    # print('fax:' in par.lower())
    ls = list(set(par))
    for j in ls:
        try:
            if 'fax:' in j.lower():
                # print(j)
                x = fax_helper(j.lower())
                # print("out is ",x)
                return (x)
                break
        except:
            pass
            # print("err ",j)


def phone_helper(st):
    id = st.find('phone'.lower())
    ctr = 0
    start = 0
    end = 0
    flg = 0
    for i in range(id + 5, len(st)):
        if ctr == 10:
            break
        if st[i].isnumeric():
            if flg == 0:
                start = i
                flg = 1
            end = i
            ctr += 1
    new = st[start:end + 1]
    return (new)


def phone(par):
    ls = list(set(par))
    for j in ls:
        try:
            if "phone".lower() in j.lower():
                if bool(re.search(r'[0-9]+', j.lower())) == True:
                    ph = phone_helper(j.lower())
                    return ph
                    break
        except:
            pass


def insured_helper(st):
    st_small = st.lower()
    start = st_small.find(":")
    end = st_small.find("chubb")
    return (st[start + 1:end])


def insured(par):
    ls = list(set(par))
    for j in ls:
        try:
            if 'Insured Name'.lower() in j.lower():
                return (insured_helper(j))
        except:
            pass


def email_helper(st):
    x = re.findall("[a-zA-Z0-9.]+@[a-zA-Z]+[.][a-zA-Z]{2,3}", st)
    return x[0]


def email(par):
    ls = list(set(par))
    for j in ls:
        try:
            if "Email".lower() in j.lower():
                if bool(re.search("[a-zA-Z0-9.]+@[a-zA-Z]+[.][a-zA-Z]{2,3}", j)) == True:
                    return (email_helper(j))
                    break
        except:
            pass


for i in range(1, 7):
    ds = pd.read_csv(f"{i}.csv")
    # print(f"This is csv {i}")

    ########       Agency Name       ######################

    out1 = agency_name(ds['new_col'])
    if type(out1) == str:
        main.loc[i - 1, 'Agency Name'] = out1.strip()
    else:
        main.loc[i - 1, 'Agency Name'] = out1

    ########     Legacy Chubb Agency Code       ###########

    out2 = agency_code(ds['new_col'])
    main.loc[i - 1, 'Legacy Chubb Agency Code'] = out2

    ##########     Chubb Ltd Policy No         ############

    out3 = Chubb_Policy_No(ds['new_col'])
    main.loc[i - 1, 'Chubb Ltd Policy No'] = out3

    ##########          STREET        #####################

    out4 = street(ds['new_col'])
    main.loc[i - 1, 'Street'] = out4

    #########            FAX       ########################

    out5 = fax(ds['new_col'])
    main.loc[i - 1, 'Fax'] = out5

    #########           PHONE      ########################

    out6 = phone(ds['new_col'])
    main.loc[i - 1, 'Phone'] = out6

    #########       Insured Name    #######################

    out7 = insured(ds['new_col'])
    main.loc[i - 1, 'Insured Name'] = out7

    #########          Email        ######################

    out8 = email(ds['new_col'])
    # print("email ", out8)
    main.loc[i - 1, 'Email'] = out8

print(main.iloc[:][0:6])

main.to_csv("StandardTemplate_aash.csv")



