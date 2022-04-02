
import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

import hashlib
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import nltk
from nltk.tokenize import ToktokTokenizer
tokenizer = ToktokTokenizer()
from nltk.tokenize import word_tokenize
from collections import Counter
import seaborn as sns 
import matplotlib.pyplot as plt  
#%matplotlib inline
from string import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import plotly.express as px
from wordcloud import WordCloud
import time
import tracemalloc
from sklearn.preprocessing import MultiLabelBinarizer

def dataEngineering(tesseract_exe_path,wordcount_plot = True):
    """
    Description: 

    All the insurance & non insurance pdf files should be stored inside the sub directory of the current working directory with the name 
    
    train_data_pdfs(name of the sub directory).

    All the insurance pdfs should be stored in the sub directory of the current working directory with the name
    
    insurance(name of the sub directory) - This is used to count the most occuring words in all the pdfs.

    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path

# current directory

    cpath = os.getcwd()
    print(f"Current Working Directory  :  \n{cpath}\n")
    
# to fetch the list of subdirectories in the current working directory

    directory_contents = os.listdir(cpath)
    #print(directory_contents)
    
# to get the path of subdirectory in which the pdf data files are stores - to work - for DataEngineering part.

    litem = []
    for item in directory_contents:
        if os.path.isdir(item) and bool(re.search("pdfs",item)):
            litem.append(item)

    len_litem = []
    for x in range(len(litem)):
        len_litem.append(len([f.name for f in os.scandir(cpath+"\\"+litem[x]) if f.is_file()]))

    s_len_item = sorted(len_litem)
    
    litem_in = len_litem.index(s_len_item[-1])
    
# sub dir path where the pdfs raw data is saved 

    path_DE = cpath+"\\"+litem[litem_in]
    
    print(f"Path in which the pdf files were stored :\n{path_DE}\n")
    
    print("-"*125)
    
# preprocessing raw data    
    print("\nPreProcessing of Data\n")
    print("-"*125)
# to create sha256 encryption for all the pdfs

    sha256 = hashlib.sha256()
    list_train = []
    encryptfile= []
    for (root,dirs,file) in os.walk(path_DE):
        for f in file:
            if ".pdf" in f:
                list_train.append(f)
                with open(path_DE+'\\'+f, 'rb') as opened_file:
                    for line in opened_file:
                        sha256.update(line)
                    encryptfile.append('{}'.format( sha256.hexdigest()))

    dict_df = {"Pdf_Name":list_train,"Encrypt_sha256":encryptfile}
    #print(dict_df)

    
# Converting pdfs to image to text

    def image_2_text(path):
        images = convert_from_path(path_pdf)
        for i in range(1):
            images[i].save(path_DE+'\\page'+str(i)+'.jpg','JPEG')
            image = path_DE+'\\page'+str(i)+'.jpg'
            text = pytesseract.image_to_string(Image.open(image),lang = "eng")
            text = text.lower()
        return text
        
# Generating all the text from the insurance pdfs and concatenating them

    #directory_contents = os.listdir(cpath)
    for item in directory_contents:
        if os.path.isdir(item) and bool(re.search("Insurance",item)) :#and bool(re.search("pdfs",item)):
            ins = item
    insurance_ = cpath+"\\"+ins
    #insurance_
    
    #insurance_ = 'C:\\Users\\SivaShankarErumala\\Downloads\\hashlib\\Insurance'

    text_img = ""

    for (root,dirs,file) in os.walk(insurance_):
        for f in file:
            if ".pdf" in f:
                #print(f)
                path_pdf = insurance_+'\\'+f

                text_img += f"\n{image_2_text(path = path_pdf)}"
    #text_img     


# PreProcessing using NLP Approach

# Removing Special Characters

    def remove_special_characters(text):
        # define the pattern to keep
        pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
        return re.sub(pat, '', text)

    txt1=remove_special_characters(text_img)
    
    
#Removing Numbers

    def remove_numbers(text):
        # define the pattern to keep
        pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
        return re.sub(pattern, '', text)

    txt2=remove_numbers(txt1)

    
# Removing Punctuation

    def remove_punctuation(text):
        text = ''.join([c for c in text if c not in string.punctuation])
        return text
    
    txt3=remove_punctuation(txt2)


# Removing Stopwords

    stopword_list=stopwords.words('english')
    
    def remove_stopwords(text):
        # convert sentence into token of words
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        # check in lowercase 
        t = [token for token in tokens if token.lower() not in stopword_list]
        text = ' '.join(t)    
        return text
       
    txt4=remove_stopwords(txt3)


# Removing Extra WhiteSpaces and Tabs

    def remove_extra_whitespace_tabs(text):
        #pattern = r'^\s+$|\s+$'
        pattern = r'^\s*|\s\s*'
        return re.sub(pattern, ' ', text).strip()
       
    txt_final = remove_extra_whitespace_tabs(txt4)


# Filtered Tokenized Text

    word_tokens = word_tokenize(txt_final)
    filter_text = [w for w in word_tokens if len(w)>2]
    # print(len(filter_text)
    
    
## Feature Extraction

# Word Cloud with insurance pdfs (to know the best features of insurance data

    if wordcount_plot :
        print("*"*40+"Word Cloud for common words in insurance pdfs"+"*"*40)
        print("\n")
        all_words = " ".join([text for text in filter_text])
        wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)
        plt.figure(figsize=(10,7))
        plt.imshow(wordcloud,interpolation = 'bilinear')
        plt.axis("off")
        plt.show()

# top 10 most frequent words in the insurance pdfs 

    word_freq = Counter(filter_text)
    common_words = word_freq.most_common(10)
    print("-"*125)
    w = []
    fc = []
    for x in common_words:
        w.append(x[0])
        fc.append(x[1])
    com_words = pd.DataFrame({"Words":w,"Frequency_Count":fc})
    print('\nMost common words in all the raw insurance pdfs :\n')
    display(com_words)
    print("-"*125)
    
    list_check = [x[0] for x in common_words]
   #print(list_check)

    dict_df1 = dict()
    dict_df1['match'] = []        
    dict_df.update(dict_df1)        
    #print(dict_df)

# Check List func

    def check_list(list_check):
        lc=[]
        for x in list_check:
            if re.search(x,text_img):
                lc.append(x)

        dict_df["match"].append(lc) 

# To extract the feature data from the first page of every pdf file(insurance & non insurance files)


    sha256 = hashlib.sha256()

    for (root,dirs,file) in os.walk(path_DE):
        for f in file:
            if ".pdf" in f:

                path_pdf = path_DE+'\\'+f

                text_img = image_2_text(path = path_pdf)

                with open(path_pdf, 'rb') as opened_file:
                    for line in opened_file:
                        sha256.update(line)
                    text_img = text_img+'\n{}'.format( sha256.hexdigest())

                check_list(list_check)


    df1 = pd.DataFrame(dict_df)
    
    print("\nData Frame After Feature Extraction and Before Feature Engineering\n")
    
    display(df1)
    
    print("\n\n")
    print("-"*125)

    mlb = MultiLabelBinarizer()
    a = mlb.fit_transform(df1["match"])
    b = mlb.classes_
    c = df1.index

    df2 = pd.DataFrame(data = a,columns = b , index = c)



    df_n = pd.concat([df1,df2],axis=1)
    df_n.drop(["match"], axis = 1, inplace=True)
    
    df_n.to_csv("dataset.csv")
    print("\nData Set is Saved!!!!!!\n")
    print("Data Frame After Feature Engineering   : \n")
    
    display(df_n)    
    
    print("\n")
    print("*"*126)
    print("*"*60+"END OF"+"*"*60)
    print("*"*55+"DATA ENGINEERING"+"*"*55)