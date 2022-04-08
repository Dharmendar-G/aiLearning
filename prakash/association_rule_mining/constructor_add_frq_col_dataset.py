# by using constructor adding frequency column to the dataset
class add_fre_col_data:
    col_name = input('enter col_name:')
    freq = {}
    new_col = []

    # parameterized constructor
    def __init__(self, data):
        self.freq = freq
        self.new_col = new_col
        self.data = data

    def add_col_data(self):
        for x in range(len(data[col_name])):
            sg = data[col_name].iloc[x].strip('[]').split(',')
            for x in sg:
                y = x.strip(" ''")
                if y in freq.keys():
                    freq[y] += 1
                else:
                    freq[y] = 1
        for x in range(len(data[col_name])):
            sg = data[col_name].iloc[x].strip('[]').split(',')
            sg = [x.strip("'' ") for x in sg]
            cn = []
            for x in sg:
                cn.append(freq[x])
            new_col.append(cn)
        print('the new_col list is', new_col)
        print('the new dataset is ', self.answer)

    def inserting_col(self):
        self.insertion = self.data.insert(len(self.data.columns), 'new_freq_col9', new_col)
        self.answer = self.data.head()
        print(self.answer)


