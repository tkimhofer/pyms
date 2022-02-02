# import sys
# sys.path.append('./')
# from msmate import ms_exp
import time
import msmate as ms
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
reload(ms)
# from sklearn.cluster import DBSCAN, OPTICS
import pandas as pd

# path='/Users/torbenkimhofer/Rproj/cam_cov_lcms/dat/'
# path='/Volumes/Backup\ Plus/hayley/'
path='/Volumes/Backup\ Plus/Cambridge_RP_NEG/'
path='/Users/tk2812/py/msfiles/'




d = ms.list_exp(path)
test=ms.ExpSet(path, ms.msExp)
test.read(pattern='DDA')

test.exp[1].plt_chromatogram(tmin=None, tmax=None, type=['tic', 'bpc', 'xic'], xic_mz=50, xic_ppm=10)
(test.exp[0]).xic(mz=100, ppm=10, rt_min=None, rt_max=None)



for x in test.exp:
    x.plt_chromatogram()


test.exp



exp = ms.msExp(d.files[3])



dd = ms.ms_exp(path, fending='mzML')

dd.read_exp(fidx=0, mslevel='0', print_summary=True)
dd.plt_chromatogram()


rr=dd
rr.read_exp(fidx=1, mslevel='0', print_summary=True)
rr.plt_chromatogram()



selection={'mz_min':0, 'mz_max':1530, 'rt_min': 50, 'rt_max': 100}
rr.vis_spectrum(q_noise=0.95, selection=selection)
dd.vis_spectrum(q_noise=0.95, selection=selection)

import time

selection_all={'mz_min':None, 'mz_max':None, 'rt_min': None, 'rt_max': None}
pp_start  = time.time()
dd.peak_picking(st_adj=6e2, q_noise=0.95, dbs_par={'eps': 0.02, 'min_samples': 2}, selection=selection, plot=True)
pp_end = time.time()

print(pp_end - pp_start)


fig, ax = dd.vis_features(selection=selection, rt_bound=0.51, mz_bound=0.1)
ax.scatter(dd.fs.rt_maxI, dd.fs.mz_maxI)

dd.find_isotopes()

len(dd.isotopes)
gz1=dd.isotopes[3]
dd.vis_ipattern( gz1, b_mz=4, b_rt=30, rt_bound=0.51, mz_bound=0.2)
dd.vis_ipattern(gz1, b_mz=10, b_rt=10, rt_bound=0.51, mz_bound=0.2)

# min_samples: min number of core or neighbour samples reachable from a sample in order to be core point, including itself
# non-core samples are samples that are also density reachable but not have min_samples neighbours

# for ms:
# min_samples: exceedes minimum number of feature data points
# eps: distance value that allows to reach min-samples-1 (self) from position
# if mz_accuracy is 0.001, st spacing=1, signal is n_st=10 and min_samples=5
# to reach 4 samples: eps= 0.001*4, with st spacing normalised such that n_st=0.001: x_st / 1/0.001
mz_acc = np.abs(.16557 - .1680)
st_spacing = 1
min_samples = 5
eps = (mz_acc * (min_samples-1))*1.1
adj = (st_spacing / (mz_acc)) * 5.1

out.peak_picking(selection=selection, dbs_par={'eps': eps, 'min_samples': min_samples-1}, st_adj=adj, q_noise=0.6)
out.vis_features(selection=selection, rt_bound=0.021, mz_bound=0.1)


selection={'mz_min':500, 'mz_max':1000, 'rt_min': 5*60, 'rt_max': 9*60}
out.peak_picking(selection=selection, dbs_par={'eps': eps, 'min_samples': min_samples}, st_adj=adj*2, q_noise=0.95)

out.find_isotopes()
len(out.isotopes)

gz1=out.isotopes[4]
out.vis_ipattern( gz1, b_mz=4, b_rt=30, rt_bound=0.04, mz_bound=0.2)


import sqlite3
sql_connect = sqlite3.connect('/Users/tk2812/Downloads/MassBank.sql')

cursor = sql_connect.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor. fetchall())



query = "SELECT * FROM factbook;"

#plt.scatter(out.scanid, out.mz)
#plt.plot(out.fs.ppm)



# automate dbs parameterisation
selection={'mz_min':625, 'mz_max':632, 'rt_min':205, 'rt_max':801}
out.vis_spectrum(q_noise=0.9, selection=selection)


out=dd
X=out.Xf
plt.scatter(X[:,0], X[:,1], c=X[:,2])


d_mz = 0.0001
d_st = 5
st_adj = d_st/(d_mz)

X1 = X.copy()
X1[:, 0] = X1[:, 0] / (st_adj*2)
dbs = OPTICS(eps=d_mz, min_samples=3, metric='euclidean').fit(X1)

plt.scatter(X[:,0], X[:,1], c=dbs.labels_)

idx=np.where(dbs.labels_==100)
plt.scatter(X[idx,0], X[idx,1], c=dbs.labels_[idx])


dbs = DBSCAN(eps=d_mz, min_samples=5).fit(X)
print(np.unique(dbs.labels_))

d_mz_a = np.linspace(0.0001, 0.1, 100)
st_adj_a = np.linspace(1, 2000, 200)


import hdbscan as hdb
cobj = hdb.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
    metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

cobj.fit(X[:,0:2])



# create the mesh based on these arrays
pp=np.array(np.meshgrid(d_mz_a, st_adj_a)).T.reshape(-1,2)
plt.scatter(pp[:,0], pp[:,1])

X=X[:,0:2]
Xreserve=X
plt.scatter(X[:,0], X[:,1], c=tt[:,2])

ncl=[]
for i in range(pp.shape[0]):
    print(i)
    X1=X.copy()
    X1[:, 0] = X1[:, 0] / 1e9 # 2.1 e-6
    dbs = DBSCAN(eps=0.0015, min_samples=10).fit(X1)
    # dbs = OPTICS(eps=pp[i,0], min_samples=30).fit(X1)

    ncl.append(len(np.unique(dbs.labels_)))

plt.plot(ncl)
plt.scatter(X[:,0], X[:,1], c=dbs.labels_)

X=out.window_mz_rt(out.Xraw, selection)

# get distances of feature in mz dimension

d_mz = np.abs(.37261 - .37749)
d_st = 4
st_adj = (d_st/(d_mz)) # the higher adjustment factor, the closer together are scantimes
print(st_adj)

d_mz=0.01
st_adj=600
out.peak_picking(selection=selection, dbs_par={'eps': d_mz, 'min_samples': 3}, st_adj=st_adj,  q_noise=0.8)
out.vis_features(selection=selection)

out.find_isotopes()
vis_ipattern(self, fgr, b_mz=10, b_rt=10, rt_bound=0.51, mz_bound=0.001)


# visualise selection of peak picked data with isotopic patterns
# define class


# synsthetic data
from scipy.stats import norm

e.rvs(10)
signl=300
noise=0.003
e=norm(signl, 5)

signal= np.concatenate([e.rvs(20), signl+norm(0, noise).rvs(10), e.rvs(10)])
signal= np.concatenate([e.rvs(20), np.ones(10)*signl, e.rvs(10)])

x_sig=np.arange(40)

plt.scatter(x_sig, signal)

C = np.array([x_sig, signal]).T

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances as ed
# min_samples: min number of core or neighbour samples reachable from a sample in order to be core point, including itself
# non-core samples are samples that are also density reachable but not have min_samples neighbours

# for ms:
# min_samples: exceedes minimum number of feature data points
# eps: distance value that allows to reach min-samples-1 (self) from position
# if mz_accuracy is 0.001, st spacing=1, signal is n_st=10 and min_samples=5
# to reach 4 samples: eps= 0.001*4, with st spacing normalised such that n_st=0.001: x_st / 1/0.001


C1=C.copy()
C1[:,0]=C[:,0]/2

plt.scatter(C1[:, 0], C1[:,1])
plt.scatter(C[:, 0], C[:,1])

3/2
C=C1
db = DBSCAN(eps=3, min_samples=7, algorithm='brute').fit(C1)
plt.scatter(C1[:, 0], C1[:,1], c=db.labels_)
plt.scatter(C1[db.core_sample_indices_, 0], C1[db.core_sample_indices_,1])

db.components_
db.n_features_in_

idx=np.where(db.labels_==0)[0]
dd=ed(C[idx,:], C[idx,:])
plt.imshow(dd)

np.triu(dd)
# min_samples: core points
# for straight line with d 1: if n_samples:
# 4, 5 (4/5 cores + 2 neigh), then min_d must be >=2
# 6 core sapmple, then min_d must be >=2





plt.scatter(C[:,0], C[:,1], c=db.labels_)

plt.scatter(pl1.mz_mean.values, pl1.mz_mean.values - pl1.iloc[i].mz_mean)




pl1.rt_min


def peak_picking(mz, I, df, st_adj=6e2, q_noise=0.89):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import DBSCAN

    scans=np.concatenate([np.ones(len(mz[i]))*df.scan_id.values[i] for i in range(len(mz))])
    stime=np.concatenate([np.ones(len(mz[i]))*df.scan_start_time.values[i] for i in range(len(mz))])
    mmz=np.concatenate(mz)
    Ic=np.concatenate(I)

    # define X
    X=np.array([scans/st_adj, mmz, Ic]).T

    # remove noise points
    thres=np.quantile(X[...,2], q=q_noise)
    X1=X[Ic>thres, ...]
    # plt.scatter(X1[...,0], X1[...,1], c=X1[...,2], s=2)


    # reduce size of matrix for testing
    # idx=np.where((X1[...,1]>418) & (X1[...,1]<828) & \
    #              ((X1[...,0]) > 4.5) & ((X1[...,0] )  < 10.8 ))[0]
    # X2=X1[idx,...]
    # plt.scatter(X2[...,0], X2[...,1])
    X2=X1

    dbs = DBSCAN(eps=0.02, min_samples=2)
    dbs.fit(X2[...,0:2])
    res=np.unique(dbs.labels_, return_counts=True)
    print('Number of clusters: ' + str(len(res[0])))

    # check if feature intensity are gaussian-like
    fs = feat_summary(dbs, X2, st_adj, st_len_min=10, plot=False, mz_bound=0.001, rt_bound=3)
    fs = est_intensities(X2, dbs, fs)

    return fs



print(np.sort(ff.ppm))
print(np.where(ff.st_span > 0))
# plt.scatter(ff.n, ff.ppm, c=ff.st_span)
plt.scatter(ff.mz_min, ff.mz_max, c=np.log(ff.st_span), s=1)
plt.scatter(ff.ppm, ff.mz_mean, c=np.log(ff.st_span), s=1)

ff=ff[ff.n > 10]

ff[ff.mz_max-ff.mz_min < 10]


ff.iloc[np.where((mmz> 600) & (mmz < 720))[0]]

iix=np.where((mmz1> 600) & (mmz1 < 720))[0]
plt.scatter(X1[1, iix],X1[0,iix],  c=dbs.labels_[iix], s=1)


fn=ff.iloc[np.where((ff.st_span )>10 & (ff.elut <20))[0],]

fn.iloc[np.argsort(fn.ppm)].iloc[0:4].T

# summaryise each cluster
def cl_summary(X, cl_id=7, st_adj=st_adj):
    # X is nx3 array with dim 2: scatime/adj, m/z, int
    idx=np.where(dbs.labels_ == cl_id)[0]
    mz_min=np.min(X[idx,1])
    mz_max=np.max(X[idx, 1])
    mz_mean = np.mean(X[idx, 1])
    dppm= (np.std(X[idx, 1]) / mz_mean) * 1e6

    rt_mean = np.mean(X[idx,0]*st_adj)
    rt_min = np.min(X[idx,0]*st_adj)
    rt_max = np.max(X[idx,0]*st_adj)

    # st_min = np.min(X[1,idx])
    # st_max = np.max(X[1, idx])

    od ={ 'n': len(idx), \
          'mz_mean': round(mz_mean, 4) , \
          'rt_mean': rt_mean, \
          'st_span': len(np.unique(X[idx,0])),\
          'ppm': np.round(dppm), 'mz_min': np.round(mz_min, 4), 'mz_max': np.round(mz_max, 4),\
          'elut':rt_max - rt_min, 'rt_min': rt_min, 'rt_max':rt_max}

    return od







# m/z bin data
acc=0.1 # dev in ppm
mzra=np.array([[np.max(x), np.min(x)] for x in mz])
mzmin=np.min(mzra[:1])-1
mzmax=np.max(mzra[:1])+1

# bin m/z and reshape into matrix
bins = np.linspace(mzmin, mzmax, round((mzmax-mzmin)/acc))
X=np.zeros((len(bins), df.shape[0]))
for i in range(len(mz)):
    out = np.digitize(mz[i], bins)
    doub=np.unique(out, return_counts=True) # counts for each bind
    lev=np.where(doub[1]>1)[0] # where count > 1
    lev_un=np.unique(out) # unique bins for s
    Imzn=np.zeros(len(np.unique(out))) # new intensity vector
    Imzn[np.where(doub[1]==1)[0]]=I[i][np.where(doub[1]==1)[0]]

    for j in lev:
        idx=np.where(out == doub[0][j])[0] # lev
        idx_un=np.where(lev_un == doub[0][j])[0]
        Imzn[idx_un]=np.max(I[i][idx])

    X[lev_un, i]=Imzn

np.max(mz[i])


out=np.zeros((8, len(mz[6])))
c=0
r=0
for j in [-4, -3, -2, -1, 1, 2, 3, 4]:
    for i in range(len(mz[6])):
        out[c,i]=np.sort(np.abs(mz[6-j]-mz[6][i]))[0]
        r +=1
    c+=1



plt.imshow(np.log(X))
plt.plot(np.sum(X, 0)) # tic
plt.plot(np.max(X, 0)) # bpc

plt.plot(np.sort(X))
thres=np.quantile(X[X>0], q=0.99)

np.sum(X>0)/np.sum((X >=0))
np.sum(X>thres)/np.sum((X >=0))

X[X<=thres]=0



from sklearn.cluster import DBSCAN

dbs = DBSCAN(eps=3, min_samples=10)
dbs.fit(X)

clustering.labels_

plt.plot(np.sum(X, 1))


np.max([np.max(x) for x in mz])

plt.plot(mz[0])
