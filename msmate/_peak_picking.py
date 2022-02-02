def est_intensities(fs, cl_membership, X2):
    from scipy.stats import norm,multivariate_normal
    import numpy as np
    import pandas as pd
    a=[]
    for i in range(fs.shape[0]):
        f = fs.cl_id.iloc[i]
        x=X2[np.where(cl_membership==f)[0],0]
        xra = np.max(x)- np.min(x)
        xs=(x/xra)
        xmean=np.mean(xs)
        xs=xs-xmean

        y=X2[np.where(cl_membership==f)[0],2]
        ymin = np.min(y)
        ymax=np.max(y-ymin)
        ys=(y-ymin)/ymax

        est = norm.pdf(xs, 0, 0.15)[..., np.newaxis]
        coef = (np.linalg.inv(est.T @ est) @ est.T) @ ys[..., np.newaxis]
        mvr=multivariate_normal(est[...,0], np.eye(len(est[...,0])))

        ynew=((est * coef)*ymax) + ymin
        #xnew = (xs + xmean) * xra

        yss=ys/np.sum(ys)

        a.append([np.sum(y), np.sum(ynew), -np.log(mvr.pdf(yss))/len(ys), np.abs(np.sum(y)-np.sum(ynew))/np.sum(y)])
        #
        # plt.plot(xnew, ynew)
        # plt.plot(x,y)

    aa=np.array(a).T

    fs['int_integral']=aa[0]
    fs['int_gaussian'] = aa[1]
    #ff['gaussian_score_lp'] = aa[2]
    fs['relerr_integral_gaussian'] = aa[3]

    return fs

def cl_summary(cl_membership, X2, cl_id=7):
    # cluster summary
    # X is data matrix
    # dbs is clustering object with lables
    # cl_id is cluster id in labels
    # st_adjust is scantime adjustment factor
    import numpy as np
    idx=np.where(cl_membership == cl_id)[0]
    # add mz and rt where I is max
    # validate if cluster contains only a single peak or if there is peak overlap: qc score of cluster!
    idx_maxI = np.argmax(X2[idx,2])
    mz_maxI = X2[idx[idx_maxI], 1]
    mz_min=np.min(X2[idx,1])
    mz_max=np.max(X2[idx, 1])
    mz_mean = np.mean(X2[idx, 1])
    ppm= (np.std(X2[idx, 1]) / mz_mean) * 1e6

    rt_mean = np.mean(X2[idx,3])
    rt_min = np.min(X2[idx,3])
    rt_max = np.max(X2[idx,3])
    rt_maxI = X2[idx[idx_maxI], 3]

    # st_min = np.min(X[1,idx])
    # st_max = np.max(X[1, idx])

    od ={ 'n': len(idx), \
          'mz_maxI': mz_maxI , \
          'rt_maxI': rt_maxI, \
          'st_span': len(np.unique(X2[idx,0])),\
          'ppm': np.round(ppm), 'mz_min': np.round(mz_min, 4), 'mz_mean': np.round(mz_mean, 4), 'mz_max': np.round(mz_max, 4),\
          'telut': np.round(rt_max - rt_min), 'rt_min': np.round(rt_min, 1), 'rt_mean': np.round(rt_mean, 1), 'rt_max': np.round(rt_max, 1)}

    return od

def feat_summary(X2, cl_membership, st_len_min=10, plot=True, mz_bound=0.001, rt_bound=3):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    import pandas as pd

    idx = np.where(cl_membership > 0)[0]
    lab_count = np.unique(cl_membership[idx], return_counts=True)

    cl_n_above = lab_count[0][np.where(lab_count[1] > st_len_min)[0]]
    cl_n_below = lab_count[0][np.where(lab_count[1] < st_len_min)[0]]

    if plot:
        cm = plt.cm.get_cmap('gist_ncar')
        fig, ax = plt.subplots()

        idx=np.where(cl_membership <= 0)[0]
        ax.scatter(X2[idx,3], X2[idx, 1],  s=0.5, c='gray', alpha=0.2)

        idx=np.where(np.isin(cl_membership, cl_n_below))[0]
        ax.scatter(X2[idx,3], X2[idx, 1], c='gray', s=0.2, alpha=0.8)

    feat=[]
    for i in cl_n_above:
        if i >= 0:
            f1 = cl_summary(cl_membership, X2, i)
            f1['cl_id'] = i
            feat.append(f1)

            if plot:
                ax.add_patch(Rectangle(((f1['rt_min']-rt_bound), f1['mz_min']-mz_bound), \
                                       ((f1['rt_max']) - (f1['rt_min']))+rt_bound*2, \
                                       (f1['mz_max'] - f1['mz_min'])+mz_bound*2, fc='gray', alpha=0.5,\
                                       linestyle="dotted"))


    if plot:
        idx=np.where(np.isin(cl_membership, cl_n_above))[0]
        im = ax.scatter(X2[idx,3], X2[idx, 1], c=np.log(X2[idx,2]), s=5, cmap=cm)
        fig.colorbar(im, ax=ax)
        fig.show()

    if len(feat) == 0:
        raise ValueError('No L1 features found')

    ff = pd.DataFrame(feat)
    return ff.sort_values('ppm')

def peak_picking(self, st_adj=6e2, q_noise=0.89, dbs_par={'eps': 0.02, 'min_samples': 2},
                 selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                 plot=False):
    import numpy as np
    from sklearn.cluster import DBSCAN

    self.st_adj = st_adj
    self.q_noise = q_noise
    self.dbs_par = dbs_par
    self.selection = selection

    # define X: adjusted scanid, mz, intensity, scantime
    self.X = np.array([self.scanid / self.st_adj, self.mz, self.I, self.scantime]).T

    # if desired reduce mz/rt regions for ppicking
    self.Xf = self.window_mz_rt(self.X, self.selection, allow_none=True, return_idc=False)

    # remove noise points for clustering
    self.noise_thres = np.quantile(self.Xf[..., 2], q=self.q_noise)

    # add new dim for noise and c membership
    noise=np.ones(self.Xf.shape[0])
    noise[np.where(self.Xf[..., 2] > self.noise_thres)[0]] = 0
    self.Xf=np.c_[self.Xf, noise]
    X2 = self.Xf[noise==0, ...]
    #X2[:,0] = X2[:,0]/
    print(f'Clustering X, shape: {X2.shape}')

    #plt.scatter(X2[:,0], X2[:,1])
    #oo= self.vis_spectrum(q_noise=self.q_noise, selection=selection)
    #return(X2)
    # perform clustering
    dbs = DBSCAN(eps=self.dbs_par['eps'], min_samples=self.dbs_par['min_samples'], algorithm='auto').fit(X2[..., 0:2])
    self.cl_membership = dbs.labels_

    cl_mem=np.ones(self.Xf.shape[0])*-1
    cl_mem[noise==0]=dbs.labels_
    self.Xf = np.c_[self.Xf, cl_mem]
    res = np.unique(self.cl_membership, return_counts=True)
    print('Number of L1 clusters: ' + str(len(res[0])))

    # check if feature intensity are gaussian-like
    fs = feat_summary(X2, self.cl_membership, st_len_min=10, plot=plot, mz_bound=0.001, rt_bound=3)
    print('Number of L2 clusters: ' + str(fs.shape[0]))

    self.fs = est_intensities(fs, self.cl_membership, X2)
    # return fs
