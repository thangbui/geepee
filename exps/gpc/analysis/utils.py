import numpy as np

def find_rank(arr):
    a = np.array(arr)
    r = np.array(a.argsort().argsort(), dtype=float)
    f = a==a
    for i in xrange(len(a)):
        if not f[i]: 
            continue
        s = a == a[i]
        ls = np.sum(s)
        if ls > 1:
            tr = np.sum(r[s])
            r[s] = float(tr)/ls
            f[s] = False
    return r

if __name__ == '__main__':
    print find_rank([1, 2, 3])
    print find_rank([1, 3, 2])
    print find_rank([1, 3, 1])
    print find_rank([1, 1, 1])
