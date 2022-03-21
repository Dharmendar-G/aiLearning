def check_vowels(letter):
    dic,sets={},set()
    vowels=['a','e','i','o','u']
    for var in letter:
        if var.lower() in vowels:
            if var in dic:
                dic[var]+=1
            else:
                dic[var]=1
        else:
            sets.update(var)
    return print('Vowels and their counts are ',dic,'\nConsonents & others are:',sets)
check_vowels(input('Plz enter your string'))