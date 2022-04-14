
#function for adding a  new frequency column to the dataset
def add_freq_col_dataset(data):
    col_name = input('enter col_name:')
    freq = {}
    for x in range(len(data[col_name])):
        sg = data[col_name].iloc[x].strip('[]').split(',')

        for x in sg:
            y = x.strip(" ''")
            if y in freq.keys():
                freq[y] += 1
            else:
                freq[y] = 1

    new_col = []
    for x in range(len(data[col_name])):
        sg = data[col_name].iloc[x].strip('[]').split(',')
        sg = [x.strip("'' ") for x in sg]
        cn = []
        for x in sg:
            cn.append(freq[x])
        new_col.append(cn)
    data.insert(len(data.columns), 'new_freq_col', new_col)
    return data.head()