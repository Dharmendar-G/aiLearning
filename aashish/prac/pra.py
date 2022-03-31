def con(st):
    dic={"ten":10,"eleven":11,"twelve":12}
    try:
        print(dic[st.lower()])
    except:
        print("please pass a valid word")
con("Twelve")