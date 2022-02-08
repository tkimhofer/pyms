import scipy.signal


def est_intensities(fs, cl_membership, X2):
    from scipy.stats import norm, multivariate_normal
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

def cl_summary(self, cl_id=7):
    # cluster summary
    # X is data matrix
    # dbs is clustering object with lables
    # cl_id is cluster id in labels
    # st_adjust is scantime adjustment factor

    # Xf has 7 columns: scanId, mz, int, st, noiseBool, st_adj, clMem

    import numpy as np
    from scipy.stats import norm, multivariate_normal
    from scipy.signal import find_peaks, peak_prominences
    from scipy import integrate


    # import matplotlib.pyplot as plt
    # cl_n_above
    # cl_id=1

    # intensity weighted mean of st
    def wImean(x, I):
        w = I / np.sum(I)
        wMean_val = np.sum(w * x)
        wMean_idx = np.argmin(np.abs(x - wMean_val))
        return (wMean_val, wMean_idx)

    def ppm_mz(mz):
        return np.std(mz) / np.mean(mz) * 1e6
    # test = np.array([1, 2, 4, 6, 43, 74, 75, 80, 83, 96, 98])\

    idx=np.where(self.Xf[:,6] == cl_id)[0]
    x = self.Xf[idx, 3]
    y = self.Xf[idx, 2]

    a_raw = integrate.trapz(y, x)

    ppm = ppm_mz(self.Xf[idx, 1])

    # minor smoothing:
    ysm = np.convolve(y, np.ones(3) / 3, mode='valid')
    xsm = x[1:-1]
    idxsm = idx[1:-1]
    a_sm = integrate.trapz(ysm, xsm)

    y_max = np.max(y)
    y = y / y_max

    ysm_max = np.max(ysm)
    ysm = ysm / ysm_max

    # bline correction
    y_bl = ((np.min(ysm[-3:]) - np.min(ysm[0:3])) / (xsm[-1] - xsm[0])) * (xsm - np.min(xsm)) + np.min(ysm[0:3])

    ybcor = ysm - y_bl
    bcor_max = np.max( ybcor*ysm_max)

    # signal to noise
    scans = np.unique(self.Xf[idx, 0])
    scan_id = np.concatenate([np.where((self.Xf[:, 0] == x) & (self.Xf[:,4] == 1))[0] for x in scans])
    noiI = np.mean(self.Xf[scan_id,2])
    sino = bcor_max / noiI

    # general feature descriptors
    idx_maxI = np.argmax(ybcor)
    mz_maxI = self.Xf[idx[idx_maxI + 1], 1]
    mz_min = np.min(self.Xf[idx, 1])
    mz_max = np.max(self.Xf[idx, 1])
    mz_mean = np.mean(self.Xf[idx, 1])

    rt_maxI = self.Xf[idx[idx_maxI + 1], 3]
    rt_mean = np.mean(self.Xf[idx, 3])
    rt_min = np.min(self.Xf[idx, 3])
    rt_max = np.max(self.Xf[idx, 3])

    # create subset with equal endings
    _, icent = wImean(xsm, ybcor)
    # if weighted mean is at borders, then peak is not fully captured, these clusters are excluded here
    if (icent < 5) or (icent > len(idxsm)-5):
        return self.feat.update({'id:'+str(x) : {'flev': 1, \
                'crit': 'peak not centered (wImean)',
              'mz_maxI': np.round(mz_maxI, 4), \
              'rt_maxI': np.round(rt_maxI, 1), \
              'int_sm': np.round(a_sm),\
              'int_raw': np.round(a_raw),\
              'st_span': np.round(rt_max - rt_min, 1), \
              'ppm': np.round(ppm), 'mz_min': np.round(mz_min, 4), 'mz_mean': np.round(mz_mean, 4),
              'mz_max': np.round(mz_max, 4), \
              'rt_min': np.round(rt_min, 1), 'rt_mean': np.round(rt_mean, 1),
              'rt_max': np.round(rt_max, 1)}})



    d = np.min([icent, np.abs(icent - len(idxsm))])
    ra=range((icent-d), (icent+d))

    xr=xsm[ra]
    yr=ybcor[ra]
    yr_max = np.max(yr)
    yr =  yr / yr_max

    # fit normal, check residual contribution outside of x window to detect additional peaks
    # y_est = norm.pdf(xsm, np.mean(xr), np.std(xr)/2)[..., np.newaxis]
    # y_norm = y_est / np.max(y_est)
    # # get original scaling
    # y_norm = y_norm * ysm_max
    # rmse = np.sum((y_norm - ybcor) ** 2) / len(idxsm)

    # log prob
    # mvr = multivariate_normal((ysm_max*ybcor), 50 * np.eye(len(y_norm[..., 0])))
    # lp = np.sum(mvr.logpdf(y_norm))
    # plt.plot(x[1:-1], np.convolve(ybcor*y_max, np.ones(3) / 3, mode='valid'))

    # from scipy.interpolate import interp1d
    # function = interp1d(x, ybcor*y_max)

    # plt.plot(x, y*y_max)
    # plt.plot(x, ybcor*y_max)
    ppm = ppm_mz(self.Xf[idx, 1])
    a = integrate.trapz(ybcor*bcor_max, x=xsm)

    # include symmetry and monotonicity
    pl = find_peaks(ybcor*bcor_max, distance=5)
    plle = len(pl[0])

    if plle > 0:
        pp = peak_prominences(ybcor*bcor_max, pl[0])
        if len(pp[0])>0:
            aa=str(np.round(pp[0][0]))
        else:
            aa=0
    else:
        pl = -1
        aa = -1

    idx_maxI = np.argmax(ybcor)
    mz_maxI = self.Xf[idx[idx_maxI+1], 1]
    mz_min = np.min(self.Xf[idx, 1])
    mz_max = np.max(self.Xf[idx, 1])
    mz_mean = np.mean(self.Xf[idx, 1])

    rt_maxI = self.Xf[idx[idx_maxI+1], 3]
    rt_mean = np.mean(self.Xf[idx, 3])
    rt_min = np.min(self.Xf[idx, 3])
    rt_max = np.max(self.Xf[idx, 3])

    od = {'flev': 2, \
        'np': plle, \
          'sino': np.round(sino), \
          'mz_maxI': np.round(mz_maxI, 4), \
          'rt_maxI': np.round(rt_maxI, 1), \
          'pprom': aa,
          'int_sm_bcor': np.round(a),
          'int_sm': np.round(a_sm),
          'int_raw': np.round(a_raw),
          'st_span': np.round(rt_max - rt_min, 1), \
          'ppm': np.round(ppm), 'mz_min': np.round(mz_min, 4), 'mz_mean': np.round(mz_mean, 4),
          'mz_max': np.round(mz_max, 4), \
          'rt_min': np.round(rt_min, 1), 'rt_mean': np.round(rt_mean, 1),
          'rt_max': np.round(rt_max, 1),
          'y_sm_bline': ybcor*ysm_max,
          'y_smooth': ysm * ysm_max,
          'y_raw': y * y_max,
          'y_bl': y_bl * ysm_max,
          'st': x,
          'st_sm': xsm
          }

    return od

def feat_summary(self, st_len_min=10, plot=True, mz_bound=0, rt_bound=0):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    # import pandas as pd

    self.feat = {}
    self.feat_l2 = []
    #idx = np.where(self.Xf[:,6] > 0)[0]
    # filter for minimal length

    min_le_bool = (self.cl_labels[1] >= st_len_min) & (self.cl_labels[0] != -1)
    cl_n_above = self.cl_labels[0][min_le_bool]
    self.feat.update(
        {'id:' + str(x): {'flev': 1, 'crit': 'n < st_len_min'} for x in self.cl_labels[0][~min_le_bool]})

    if plot:
        cm = plt.cm.get_cmap('gist_ncar')
        fig, ax = plt.subplots()
        ax.set_facecolor('#EBFFFF')

    # import time
    # start = time.time()
    # cc=0
    for i in cl_n_above:
        print(i)
        f1 = self.cl_summary(i)
        ufid = 'id:' + str(i)
        self.feat.update({ufid: f1})
        if f1['flev'] == 2:
            self.feat_l2.append(ufid)
            if plot:
                ax.add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
                                       ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
                                       (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='#C2D0D6',\
                                       linestyle="solid", color='red', linewidth=2, zorder=0))

    # stop = time.time()
    # print(stop - start)

    if plot:
        idx = np.where(self.Xf[:, 4] <= 0)[0]  # noise
        ax.scatter(self.Xf[idx, 3], self.Xf[idx, 1], s=0.2, c='gray', alpha=1)
        idx = np.where(np.isin(self.Xf[:, 6], cl_n_above))[0]
        im = ax.scatter(self.Xf[idx,3], self.Xf[idx, 1], c=np.log(self.Xf[idx,2]), s=5, cmap=cm)
        fig.colorbar(im, ax=ax)
        fig.show()

    if len(self.feat_l2) == 0:
        raise ValueError('No L2 features found')

    #self.ftab = pd.DataFrame(feat).sort_values('ppm')




def peak_picking(self, st_adj=6e2, q_noise=0.89, dbs_par={'eps': 0.02, 'min_samples': 2},
                 selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                 plot=False):
    import numpy as np
    from sklearn.cluster import DBSCAN

    self.st_adj = st_adj
    self.q_noise = q_noise
    self.dbs_par = dbs_par
    self.selection = selection

    # if desired reduce mz/rt regions for ppicking
    # Xraw has 4 columns: scanId, mz, int, st
    self.Xf = self.window_mz_rt(self.Xraw, self.selection, allow_none=True, return_idc=False)

    # det noiseInt and append to X
    self.noise_thres = np.quantile(self.Xf[..., 2], q=self.q_noise)
    # filter noise data points
    idx_signal = np.where(self.Xf[..., 2] > self.noise_thres)[0]
    noise=np.ones(self.Xf.shape[0])
    noise[idx_signal] = 0
    self.Xf=np.c_[self.Xf, noise]
    # Xf has 5 columns: scanId, mz, int, st, noiseBool

    # st adjustment and append to X
    st_c = np.zeros(self.Xf.shape[0])
    st_c[idx_signal] = self.Xf[idx_signal, 0] / self.st_adj
    self.Xf = np.c_[self.Xf, st_c]
    # Xf has 6 columns: scanId, mz, int, st, noiseBool, st_adj
    print(f'Clustering, number of dp: {self.Xf.shape[0]}')

    #plt.scatter(X2[:,0], X2[:,1])
    #oo= self.vis_spectrum(q_noise=self.q_noise, selection=selection)
    #return(X2)
    # perform clustering
    dbs = DBSCAN(eps=self.dbs_par['eps'], min_samples=self.dbs_par['min_samples'], algorithm='auto').fit(self.Xf[idx_signal, :][..., np.array([1, 5])])
    #self.cl_membership = dbs.labels_

    # append clMem to X
    cl_mem=np.ones(self.Xf.shape[0])*-1
    cl_mem[idx_signal] = dbs.labels_
    self.Xf = np.c_[self.Xf, cl_mem]
    # Xf has 7 columns: scanId, mz, int, st, noiseBool, st_adj, clMem
    self.cl_labels = np.unique(dbs.labels_, return_counts=True)
    print('Number of L1 features: ' + str(len(self.cl_labels[0])))

    # feature description and QC filtering
    self.feat_summary(st_len_min=10, plot=plot, mz_bound=0, rt_bound=0) # self.feat: list
    print('Number of L2 features: ' + str(len(self.feat_l2)))

    # plot here if requested
    #self.fs = est_intensities(self.ftab, self.cl_membership, self.Xf)
    # return fs


def vis_feature_pp(self, selection={'mz_min': 425, 'mz_max': 440, 'rt_min': 400, 'rt_max': 440}, rt_bound=0, mz_bound=0):
        # l2 and l1 features mz/rt plot after peak picking
        # label feature with id
        # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as np
        import pandas as pd

        def vis_feature(fdict, ax=None, add=False):
            import matplotlib.pyplot as plt

            # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
            if isinstance(ax, type(None)):
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(fdict['st'], fdict['y_raw'], label='raw', linewidth=0.5, color='black')
            ax.plot(fdict['st_sm'], fdict['y_smooth'], label='smoothed', linewidth=0.5, color='cyan')
            ax.plot(fdict['st_sm'], fdict['y_bl'], label='baseline', linewidth=0.5, color='black', ls='dashed')
            ax.plot(fdict['st_sm'], fdict['y_sm_bline'], label='smoothed bl-cor', color='orange')
            if not add:
                ax.set_title(
                    f's/n: ' + str(int(fdict['sino'])) + ', prom:' + str(fdict['pprom']) + ', np:' + str(fdict['np']))
                ax.legend()

            return ax

        # Xf has 7 columns: 0 - scanId, 1 - mz, 2 - int, 3 - st, 4 - noiseBool, 5 - st_adj, 6 - clMem

        # which feature - define plot axis boundaries
        Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)

        cm = plt.cm.get_cmap('gist_ncar')


        fig, axs = plt.subplots(2,1,gridspec_kw={'height_ratios': [1, 2]}, figsize=(8,10))
        axs[0].set_xlabel('st [s]', fontsize=14)

        vis_feature(self.feat[self.feat_l2[0]], ax=axs[0])

        axs[1].set_facecolor('#EBFFFF')
        axs[1].text(1.27, 0, self.df.fname[0], rotation=90, fontsize=6, transform=axs[1].transAxes)
        axs[1].scatter(Xsub[Xsub[:,4]==0, 3], Xsub[Xsub[:,4]==0, 1], s=0.1, c='gray', alpha=0.5)

        coords = []
        for i in self.feat_l2:
            fid = float(i.split(':')[1])
            Xsub[Xsub[:,6] == fid, :]
            f1 = self.feat[i]

            # create rectangle coordinates for callbacks
            coords.append({
                'id': i,
                'x_min': f1['rt_min'] - 1,
            'x_max' : f1['rt_max'] + 1,
            'y_min' : f1['mz_min'] - 0.3,
            'y_max' : f1['mz_max'] + 0.3})

            axs[1].add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
                                   ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
                                   (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='#C2D0D6', \
                                   linestyle="solid", color='red', linewidth=2, zorder=0))
            im = axs[1].scatter(Xsub[Xsub[:,6] == fid, 3], Xsub[Xsub[:,6] == fid, 1], c=np.log(Xsub[Xsub[:,6] == fid, 2]), s=5, cmap=cm)

        coords = pd.DataFrame(coords)

        def onclick(event):
            print(event.xdata)
            print(event.xdata.dtype)

            idx=np.where((event.xdata >= coords.x_min.values) & (event.xdata <= coords.x_max.values) & (event.ydata <= coords.y_max.values) & (event.ydata >= coords.x_min.values))[0]
            print(idx.__class__)
            print(idx)
            print(idx.dtype)
            idx=list(idx)
            print(idx)
            if len(idx) > 0:
                if len(idx) > 1:
                    iid = np.argmin(np.abs(event.ydata - coords.y_max.iloc[idx].values))
                    idx = idx[iid]
                else: idx = idx[0]
                if event.dblclick == 0:
                    axs[0].clear()
                    add = False
                else:
                    add = True
                vis_feature(self.feat[coords.id.iloc[idx]], ax=axs[0], add=add)

        fig.colorbar(im, ax=axs[1], shrink=0.5)
        #fig.colorbar(pcm, ax=axs[0, :2], shrink=0.6, location='bottom')
        cid = fig.canvas.mpl_connect('button_press_event', onclick)


