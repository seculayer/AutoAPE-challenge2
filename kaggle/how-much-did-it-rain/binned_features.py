import pandas as pd
import numpy as np
import time
import sys

# creates features based on the average value of a variable in small time windows throughout 
# the hour of data available for each label. 
def main(argv):
    min_count = int(argv[2])
    max_count = int(argv[3])
    num_bins = int(argv[4])
    target = argv[1]
    path = argv[0]

    data = pd.read_csv(path+target+'.csv', index_col=0)
    tte = pd.read_csv(path+'TimeToEnd.csv', index_col=0)

    counts = pd.DataFrame(len(tte.columns) - tte.isnull().sum(axis=1))
    counts.columns = ['cnt']

    data = pd.concat([data,counts],axis=1)
    tte = pd.concat([tte,counts],axis=1)

    data = data.query('cnt > @min_count and cnt < @max_count')
    tte = tte.query('cnt > @min_count and cnt < @max_count')

    data = data.replace([-99900.0,-99901.0,-99903.0,999.0], np.nan)

    length_hour = 60
    bin_length = length_hour/num_bins
    bin_bounderies = []
    temp = 0
    while temp < length_hour-bin_length:
        temp += bin_length
        bin_bounderies.append(temp)

    new_feats = np.zeros([len(data),num_bins])
    for f,j in enumerate(data.index):
        #start = time.time()
        temp_data = data.iloc[f,:]
        temp_tte = tte.iloc[f,:]

        temp = pd.concat([temp_data,temp_tte],axis=1).dropna() 

        temp.columns = ['a','b']
        temp = temp.iloc[:-1,:]
        temp = temp.sort('b')

        temp_min = temp['b'].min()
        temp_max = temp['b'].max()
        for e,i in enumerate(bin_bounderies):
            if i < temp_min or i  > temp_max:
                new_feats[f,e] = -99999
                continue
            else:
                temp2 = temp[temp['b'] < i]
                if len(temp2) !=0:
                    temp2 = temp2[temp2['b'] >= i-bin_length]
               
                if len(temp2) == 0:
                    new_feats[f,e] = -99999
                else:
                    new_feats[f,e] = temp2.a.mean()

        last_bound = bin_bounderies[-1]
        temp2 = temp[temp['b'] > last_bound]
        if len(temp2) == 0:
            new_feats[f,num_bins-1] = -99999
        else:
            new_feats[f,num_bins-1] = temp2.a.mean()
    new_feats_df = pd.DataFrame(new_feats, index=data.index, columns=[target + '_' +str(x) for x in bin_bounderies+[61]])
    new_feats_df.index.name = 'Id'
    new_feats_df.to_csv(path+target+str(min_count+1)+'_'+str(max_count-1) +'.csv')

if __name__ == "__main__":
    main(sys.argv[1:])
    print 'finished ' +  sys.argv[1] +sys.argv[2] + ' ' +  sys.argv[3] + ' to ' +  sys.argv[4]








