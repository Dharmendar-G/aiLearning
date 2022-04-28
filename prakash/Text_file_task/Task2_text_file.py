import re
def fun5(filename):
    try:
        file2 = open(filename, 'r')
        for i in file2:
            if 'table' in i:
                pattern = re.findall(r'[A-Z0-9]', i)
                print("".join(pattern))
    except:
        print('file is not opening')

fun5('file2.txt')