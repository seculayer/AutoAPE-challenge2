import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression

def read(csv_dir):
    csv_file = pd.read_csv(csv_dir)
    csv_file.columns = ['id', 'shtp','rctp', 'day', 'time', 'state', 'loca', 'grsz', 'hmown','carage',
                        'carval','riskfac','old','young','married','prev','dura','a','b','c',
                        'd','e','f','g','cost']
    return csv_file

def rearrange_int(file):
    output = np.array([])
    output = pd.DataFrame({'id':file['id'],
                           'shtp':file['shtp'],
                           'rctp':file['rctp'],
                           'day':file['day'],
                           'grsz':file['grsz'],
                           'carage':file['carage'],
                           'riskfac':file['riskfac'],
                           'old':file['old'],
                           'young':file['young'],
                           'married':file['married'],
                           'prev':file['prev'],
                           'dura':file['dura'],
                           'a':file['a'],
                           'b':file['b'],
                           'c':file['c'],
                           'd':file['d'],
                           'e':file['e'],
                           'f':file['f'],
                           'g':file['g'],
                           'cost':file['cost']})
    return output

def rearrange_x(file):
    output = np.array([])
    output = pd.DataFrame({'shtp':file['shtp'],
                           'rctp':file['rctp'],
                           'day':file['day'],
                           'grsz':file['grsz'],
                           'carage':file['carage'],
                           'riskfac':file['riskfac'],
                           'old':file['old'],
                           'young':file['young'],
                           'married':file['married'],
                           'prev':file['prev'],
                           'dura':file['dura'],
                           'cost':file['cost']})
    return output

def string2int(csv_file): #only int included!

    csv_file = csv_file.fillna(0)
    print(type(csv_file.values[0][0]))
    for i in range(len(csv_file.values)):
        for j in range(len(csv_file.values[i])):
            csv_file.values[i][j] = int(csv_file.values[i][j])

    print("string2int: ")
    print(csv_file)
    return csv_file

def lr(x_train,y,x_test): #x->2d, y->1d
    reg = LinearRegression()
    reg.fit(x_train,y)
    y_predict = reg.predict(x_test)
    print(y_predict)
    return y_predict


def write_csv(x_test,y_predict):
    result = []
    for i in range(len(x_test)):
        result.append([x_test[i],y_predict[i]])

    with open('submit.csv','w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f)
        first = ['customer_ID', 'plan']
        writer.writerow(first)
        for i in range(len(result)):
            writer.writerow(result[i])

def min_max(a,b,c,d,e,f,g):
    min0 = 0
    min1 = 1
    max1 = 1
    max2 = 2
    max3 = 3
    max4 = 4
    for i in range(len(a)):
        if   a[i] < min0: a[i] = min0
        elif a[i] > max2: a[i] = max2
        if   b[i] < min0: b[i] = min0
        elif b[i] > max1: b[i] = max1
        if   c[i] < min1: c[i] = min1
        elif c[i] > max4: c[i] = max4
        if   d[i] < min1: d[i] = min1
        elif d[i] > max3: d[i] = max3
        if   e[i] < min0: e[i] = min0
        elif e[i] > max1: e[i] = max1
        if   f[i] < min0: f[i] = min0
        elif f[i] > max3: f[i] = max3
        if   g[i] < min1: g[i] = min1
        elif g[i] > max4: g[i] = max4
    print("min_max")
    print(a)
    print(b)
    return a,b,c,d,e,f,g

def concat(a,b,c,d,e,f,g):
    output=[]
    for i in range(len(a)):
        output.append(repr(round(a[i])) +
                      repr(round(b[i])) +
                      repr(round(c[i])) +
                      repr(round(d[i])) +
                      repr(round(e[i])) +
                      repr(round(f[i])) +
                      repr(round(g[i]))  )
    return output

def arrange_test_x_ori(result, test_x_ori):
    past = 0
    output = []
    for i in range(len(test_x_ori['id'])):
        if test_x_ori['id'][i] != past:
            output.append(result[i])
            past = test_x_ori['id'][i]
        else:
            continue
    print("arrange_test_x_ori")
    print(output)
    return output



if __name__ == "__main__":
    train_x_ori = read('../161_allstate/data/train.csv')
    test_x_ori  = read('../161_allstate/data/test_v2.csv' )

    train_x = rearrange_int(train_x_ori)
    test_x  = rearrange_int(test_x_ori )
    train_x = train_x.fillna(0)
    test_x = test_x.fillna(0)
    print(train_x)

    a = lr(rearrange_x(train_x), train_x['a'], rearrange_x(test_x))
    b = lr(rearrange_x(train_x), train_x['b'], rearrange_x(test_x))
    c = lr(rearrange_x(train_x), train_x['c'], rearrange_x(test_x))
    d = lr(rearrange_x(train_x), train_x['d'], rearrange_x(test_x))
    e = lr(rearrange_x(train_x), train_x['e'], rearrange_x(test_x))
    f = lr(rearrange_x(train_x), train_x['f'], rearrange_x(test_x))
    g = lr(rearrange_x(train_x), train_x['g'], rearrange_x(test_x))
    a,b,c,d,e,f,g = min_max(a,b,c,d,e,f,g)
    result = concat(a,b,c,d,e,f,g)
    output = arrange_test_x_ori(result, test_x_ori)
    x_test = list(set(test_x_ori['id']))
    print(x_test)
    write_csv(x_test,output)

