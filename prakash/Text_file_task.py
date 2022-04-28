import re
def fun2():
    try:
        file2 = open('text.txt', 'r')
        for i in file2:
            pattern = re.findall(r'[A-Z0-9]', i)
            print("".join(pattern))

    except:
        print('file is not opening')


fun2()
