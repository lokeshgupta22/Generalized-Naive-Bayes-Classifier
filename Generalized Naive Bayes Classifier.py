import numpy as np
import pandas as pd
import time
import statistics
import math
import warnings
warnings.filterwarnings("ignore")


def average(x):
    return statistics.mean(x)


def stdev(x):
    return statistics.stdev(x)


def norm(x, mean, stdev):
    ans = (1/(stdev*(math.sqrt(2*math.pi)))) * \
        (math.e**(-0.5*(((x - mean)/stdev)**2)))
    return ans


def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def accuracy(path):
    df = pd.read_csv(path)
    lencol = []
    ocn = -1
    srno = 0
    for i in range(0, len(df.columns)):
        arrr = {}
        v = df.iloc[:, i]
        v = list(v)
        for j in v:
            if j not in arrr:
                arrr[j] = 1
            else:
                arrr[j] += 1
        lencol.append(len(arrr))
    if len(df.index) in lencol:
        srno = lencol.index(len(df.index))
        df.drop(columns=df.columns[srno], axis=1, inplace=True)
        lencol.remove(len(df.index))
    ocn = lencol.index(min(lencol))

    col = df.columns
    ar = {}
    arr = {}
    for i in range(0, len(df.iloc[:, ocn])):
        val = df._get_value(i, col[ocn])
        if val not in ar:
            ar[val] = 1
        else:
            ar[val] += 1
    k = sorted(list(ar.keys()))
    if type(k[0]) == str:
        for i in range(0, len(df.iloc[:, ocn])):
            val = df._get_value(i, col[ocn])
            for j in range(len(k)):
                if val in k:
                    ind = k.index(val)
                    df.iloc[:, ocn][i] = ind
    df.set_index(df.iloc[:, ocn], inplace=True)
    df.drop(columns=df.columns[ocn], axis=1, inplace=True)

    if type(k[0]) == str:
        for i in range(0, len(df.index)):
            val = df.index[i]
            if val not in ar:
                arr[val] = 1
            else:
                arr[val] += 1
        key = list(arr.keys())
        k = key
    avg = []
    se = []
    p = []
    cr = len(df.columns)
    for j in k:
        m = 0
        p.append(len(df.loc[j])/len(df.index))
        for i in range(cr):
            list2 = df.loc[j][df.columns[m]]
            list2 = list(list2)
            avg.append(average(list2))
            se.append(stdev(list2))
            m += 1

    d = {}
    for x in range(0, len(df.index)):
        d["l{0}".format(x)] = list(df.iloc[x])

    avg = list(split(avg, cr))
    se = list(split(se, cr))
    d["avg"] = avg
    d["stdev"] = se
    d["prob"] = p
    v = []
    ans = []
    for j in k:
        for x in range(0, len(df.index)):
            val = 1
            data = d["l{0}".format(x)]
            mean = avg[j]
            dev = se[j]
            pro = p[j]
            for i in range(cr):
                val = val*norm(data[i], mean[i], dev[i])
                v.append(val)
            ans.append(val*pro)

    ans = list(split(ans, int(len(df.index))))
    maxval = []
    predict = []
    for i in range(len(df.index)):
        m = []
        for j in k:
            m.append(ans[j][i])
        maxval.append(max(m))
        ke = m.index(max(m))
        predict.append(ke)

    given = list(df.index)
    count = 0
    for i in range(len(df.index)):
        if given[i] == predict[i]:
            count += 1
    accuracy = (count)/(len(df.index))
    print(f"Accuracy of {path}: {accuracy}")


start = time.time()
path = ["Iris.csv","diabetes.csv"]
for i in path:
    accuracy(i)
print(f"Time taken: {time.time() - start}")
