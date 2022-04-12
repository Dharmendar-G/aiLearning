import matplotlib.pyplot as plt
import seaborn as sns
from get_data import get_Data
from univariate_analysis import Univariate_Analysis

class Visualization:
    def __init__(self):
        pass

    def boxplot(self, data):
        for i in list(data.columns):
            print(i)
            sns.boxplot(x = data[i])
            plt.text(max(data[i])+50,0, 'IQR =' + str(Univariate_Analysis().iqr(get_Data())[i]), bbox = dict(facecolor='pink',alpha=0.5),fontsize = 14)
            plt.text(max(data[i])+50,0.1, 'upper_limit =' + str(Univariate_Analysis.upper_range_iqr(get_Data())[i]), bbox = dict(facecolor='red',alpha=0.5), fontsize = 14)
            plt.text(max(data[i])+50,0.2, 'lower_limit =' + str(Univariate_Analysis.lower_range_iqr(get_Data())[i]),bbox = dict(facecolor='green',alpha=0.5), fontsize = 14)
            plt.text(max(data[i])+50,0.3, 'no of outliers =' + str(Univariate_Analysis.no_of_outliers(get_Data(),i)),bbox = dict(facecolor='black',alpha=0.5), fontsize = 14)
            #plt.text(max(data[i])+50,0.4, '% of outliers =' + str((no_of_outliers(i)/len(data[i])*100),bbox = dict(facecolor='purple',alpha=0.5), fontsize = 14))  
            plt.show()

    def distplot(self, data):
        for i in data.columns:
            sns.distplot(data[i], color = 'b')
            plot = plt.show()
            return plot

    