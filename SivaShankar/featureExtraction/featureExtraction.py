'''#### pending

- Number of floors in entire building (including basement/enclosed area, if any):(1 floor/2 floors/3 or more floors)

#### Done

- Do you have access to basement storage? (Yes/No)
- pattern
             basement storage? yes no []  => yes 
             
             
- Any flood or water damage losses, paid or unpaid, in the last 10 years? (Yes/No)
- pattern
			- found in 1,2,and 5 files 
			1.csv   	=> 10 years? ©oyes ©no   => no
			2.csv   	=> 10 years? oyes ©no    => no
			5th txt file    =>10 years? yes |v] no |] => yes


- Date of loss : 
- pattern
				its present in the 1.csv,2.csv,4.csv and 5 th file the output.txt
				data loss patterns:
						the key is like :
	
							date of loss or date(s) of loss
						and 
							date of loss (next to it is 'xx/xx/xxxx')  => 'xx/xx/xxxx'
							date of loss (next to it is 'Amount')	   => "null"
							date of loss (next-> ":" then the 'xx/xx/xxxx')	   => 'xx/xx/xxxx'
							date of loss (next-> ":" then the 'xx/xx/xxxx') then amount   => 'xx/xx/xxxx'

				in 5th text file =>  there are 3 date of loss values => currently updating the code on this scenario. => (Done)

				output of date of loss => if date of loss  is not filled then => null
							  if no date of loss in file => not in file
							  if nothing is there after date of loss => null
							  if date is written it checks if the string is a perfect date or not if yes "print date" else: "print null"
						null => not filled
						not in file => not in respective csv/txt file
						date xx/xx/xxxx => is perfectly filled
'''

import pandas as pd
import re
import nltk
import warnings
warnings.filterwarnings("ignore")
from dateutil.parser import parse 

def baseStorage(ss):
    c = nltk.word_tokenize(ss)
    yy = "strt"
    nn = "strt"
    bss_ = ""
    for x in range(len(c)):
        if bool(re.search("storage",c[x])):
            #print(x,c[x])
            if c[x+1]=="?":
                #print(x+1,c[x+1])
                if c[x+2] == "yes" and c[x+3] == "no":
                    yy = c[x+2]
                if c[x+3] == "no" and c[x+4]=="[" and c[x+5]=="]":
                    nn = "no not"
    bs = [yy,nn]

    bss = [x for x in bs if not(bool(re.search("not",x)))]
    #print(bss)
    if len(bss)==0:
        bss_ = "NA"
    elif len(bss)==1:
        bss_ = bss[0]   

    return bss_

def year_10(ss):
    c = nltk.word_tokenize(ss)
    yy = "start"
    nn = "start"
    year_ = ""
    for x in range(len(c)):
    #     nn = "not in file"
        if bool(re.search("years",c[x])):
            #print(x,c[x])
            if c[x+1] == "?":
                #print(x+1,c[x+1])
                if bool(re.search("yes",c[x+2])):
                    #print(c[x+2])
                    if bool(re.search("oy",c[x+2])):
                        yy = "yes not"
                    elif bool(re.search("©y",c[x+2])):
                        yy = "yes"
                    elif bool(re.search("v",c[x+3])):
                        yy = "yes"
                    elif bool(re.search("v",c[x+4])):
                        yy = "yes"
                if bool(re.search("no",c[x+3])):
                    #print(c[x+3])
                    if bool(re.search("on",c[x+3])):
                        nn = "no not"
                    elif bool(re.search("©",c[x+3])):
                       # print(x+3,c[x+3])
                        nn = "no"
                else:
                    for d in range(x+2,x+7):
                        if bool(re.search("no",c[d])):
                            #print(d,c[d])
                            for f in range(d+1,d+4):
                                if bool(re.search("v",c[f])):
                                    nn = "no"
                                else:
                                    nn = "no not"
                                     
    year = [yy,nn]
    years = [x for x in year if not(bool(re.search("not",x)))]

    if len(years)==0:
        year_ = "NA"
    elif len(years)==1:
        year_ = years[0]

    return year_

            
                

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False  

def dateloss(ss):
    c = nltk.word_tokenize(ss)
    #out = "NA"
    outs = []
    for x in range(len(c)):
        if bool(re.search("date",c[x])):
           # print(x,c[x])
            m = x
            for z in range(m,m+6):
                if bool(re.search("loss",c[z])):
                    #print(z,c[z])
                    n = z
                
                    if bool(re.search("amount",c[n+1])):
                        #print("date = null")
                        outs.append("NA")
                    if bool(re.search(r"[0-9/]",c[n+1])):
                        #print(n+1,c[n+1])
                        outs.append(str(c[n+1]))
                    if bool(re.search(r"[:]",c[n+1])):
                        if bool(re.search(r"[0-9/]",c[n+2])):
                            #print(n+2,c[n+2])
                            outs.append(str(c[n+2]))
    if len(outs) == 0:
        out = ""
    if len(outs) == 1:
        if bool(re.search(r"[/]",outs[0])):
            if bool(re.search(r"[0-9]",outs[0])):
                out = outs[0]
            else:
                out = "NA"
        else:
            out = outs[0]
    if len(outs) == 3:
        ot = []
        for ou in outs:
            if bool(re.search(r"[/]",ou)):
                if bool(re.search(r"[0-9]",ou)):
                    if is_date(ou):
                        ot.append(ou)
        if len(ot) == 0:
            out = "NA"
        else:
            out = ot[0]
    
    return out


stdf = pd.read_csv("dataset/StandardTemplate.csv", encoding='cp1252')
for xx in range(1,7):
    if xx == 5:
        f = open('dataset/5csv_complete_text.txt','r')
        ss = f.readlines()[0]
        ss = ss.lower()
        w = dateloss(ss)
        yr = year_10(ss)
        b_S = baseStorage(ss)
        if stdf["fileName"].iloc[xx-1] == xx:
            stdf["Date of loss"].iloc[xx-1] = str(w)
            stdf['Any flood or water damage losses, paid or unpaid, in the last 10 years? (Yes/No)'].iloc[xx-1] = str(yr)
            stdf['Do you have access to basement storage? (Yes/No)'].iloc[xx-1] = str(b_S)
#         print(w,"\n")
#         print(yr,"\n")
#         print(b_S,"\n")
#         print("-------------")
#         print(ss,"\n\n")
    else:
        df = pd.read_csv(f"dataset/{xx}.csv")
        z = []
        for y in df.columns:
            if bool(re.search("Unnamed:",y)):
                z.append(y)
        df.drop(z,axis=1,inplace=True)
        ss = ""
        for x in list(df.block_num.unique()):
            data = df[df["block_num"]==x]
            for y in list(data.par_num.unique()):
                dat = data[data["par_num"]==y]
                for z in list(dat.line_num.unique()):
                    da = dat[dat["line_num"]==z]
                    ss +=" ".join([str(list(da.text[da.word_num == w])[0]) for w in list(da.word_num.unique())])+" "
        ss = ss.lower()
        w = dateloss(ss)
        yr = year_10(ss)
        b_S = baseStorage(ss)
        
        if stdf["fileName"].iloc[xx-1] == xx:
            stdf["Date of loss"].iloc[xx-1] = str(w)
            stdf['Any flood or water damage losses, paid or unpaid, in the last 10 years? (Yes/No)'].iloc[xx-1] = str(yr)
            stdf['Do you have access to basement storage? (Yes/No)'].iloc[xx-1] = str(b_S)
#         print(w,"\n")
#         print(yr,"\n")
#         print(b_S,"\n")
#         print("-------------")
#         print(ss ,"\n\n")

#stdf["Date of loss"] 
# NA - not filled ; 
# empty string  -  not in the file

#stdf['Any flood or water damage losses, paid or unpaid, in the last 10 years? (Yes/No)']
# empty string - not in the file

#stdf['Do you have access to basement storage? (Yes/No)']
# empty string -  no in the file.

stdf.to_csv("SivaShankar/featureExtraction/StandardTemplateSubmission_SS.csv",index=False)
print("Done...")
#print(list(stdf.columns))

