import numpy as np
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
    # cl_id = 3660
    def wImean(x, I):
        # intensity weighted mean of st
        w = I / np.sum(I)
        wMean_val = np.sum(w * x)
        wMean_idx = np.argmin(np.abs(x - wMean_val))
        return (wMean_val, wMean_idx)

    def ppm_mz(mz):
        return np.std(mz) / np.mean(mz) * 1e6
    # test = np.array([1, 2, 4, 6, 43, 74, 75, 80, 83, 96, 98])\

    # dp idx for cluster
    idx=np.where(self.Xf[:,6] == cl_id)[0]
    iLen = len(idx)

    # gaps or doublets in successive scan ids
    x_sid = self.Xf[idx, 0]
    sid_diff = np.diff(x_sid)
    sid_gap = np.sum((sid_diff > 1) | (sid_diff < 1)) / iLen

    # import matplotlib.pyplot as plt
    x = self.Xf[idx, 3]
    y = self.Xf[idx, 2]
    mz = self.Xf[idx, 1]
    ppm = ppm_mz(mz)

    mz_min = np.min(mz)
    mz_max = np.max(mz)
    rt_min = np.min(x)
    rt_max = np.max(x)

    if ppm > self.qc_par['ppm']:
        crit = 'ppm: m/z variability'
        qc = {'sId_gap': sid_gap, 'ppm': ppm}
        quant = {}
        descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                 'mz_min': mz_min, 'mz_max': mz_max, \
                 'rt_min': rt_min, 'rt_max': rt_max, }
        fdata = {'st': x, 'I_raw': y}
        od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od

    if sid_gap > self.qc_par['sId_gap']:
        crit = 'sId_gap: scan ids missing'
        qc = {'sId_gap': sid_gap, 'ppm': ppm}
        quant = {}
        descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                 'mz_min': mz_min, 'mz_max': mz_max, \
                 'rt_min': rt_min, 'rt_max': rt_max, }
        fdata = {'st': x, 'I_raw': y}
        od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od

    _, icent = wImean(x, y)

    # minor smoothing:
    ysm = np.convolve(y, np.ones(3) / 3, mode='valid')
    xsm = x[1:-1]
    idxsm = idx[1:-1]

    # I min-max scaling
    y_max = np.max(y)
    y = y / y_max
    ysm_max = np.max(ysm)
    ysm = ysm / ysm_max

    # bline correction of smoothed signal
    y_bl = ((np.min(ysm[-3:]) - np.min(ysm[0:3])) / (xsm[-1] - xsm[0])) * (xsm - np.min(xsm)) + np.min(ysm[0:3])

    ybcor = ysm - y_bl
    bcor_max = np.max(ybcor*ysm_max)

    # check peak centre
    _, icent = wImean(xsm, ybcor)

    if (icent < 2) | (icent > len(idxsm) - 2):
        crit = 'icent: Peak is not centere'
        qc = {'sId_gap': sid_gap, 'icent': icent, 'ppm': ppm}
        quant = {}
        descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                 'mz_min': mz_min, 'mz_max': mz_max, \
                 'rt_min': rt_min, 'rt_max': rt_max, }
        fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'I_raw': y * y_max,
                 'I_bl': y_bl * ysm_max, }
        od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od


    nneg = np.sum(ybcor >= -0.05) / len(ysm)
    if nneg < self.qc_par['non_neg']:
        crit = 'nneg: Neg Intensity values'
        qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'ppm': ppm}
        quant = {}
        descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                 'mz_min': mz_min, 'mz_max': mz_max, \
                 'rt_min': rt_min, 'rt_max': rt_max, }
        fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor*ysm_max, 'I_smooth': ysm * ysm_max, 'I_raw': y * y_max, 'I_bl': y_bl * ysm_max,}
        od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od

    # signal / noise
    scans = np.unique(x_sid)
    scan_id = np.concatenate([np.where((self.Xf[:, 0] == x) & (self.Xf[:,4] == 1))[0] for x in scans])
    noiI = np.median(self.Xf[scan_id,2])
    sino = bcor_max / noiI


    if sino < self.qc_par['sino']:
        crit = 'sino: s/n below qc threshold'
        qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'ppm': ppm}
        quant = {}
        descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                 'mz_min': mz_min, 'mz_max': mz_max, \
                 'rt_min': rt_min, 'rt_max': rt_max, }
        fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'I_raw': y * y_max,
                 'I_bl': y_bl * ysm_max, }
        od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od


    # intensity variation level
    idx_maxI = np.argmax(ybcor)
    sdxdy = (np.sum(np.diff(ybcor[:idx_maxI]) < (-0.05)) + np.sum(np.diff(ybcor[(idx_maxI + 1):]) > 0.05)) / (
                len(ybcor) - 3)

    if sdxdy > self.qc_par['raggedness']:
        crit = 'raggedness: intensity variation level above qc threshold'
        qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'raggedness': sdxdy, 'ppm': ppm}
        quant = {}
        descr = {'ndp': iLen, 'st_span': np.round(rt_max - rt_min, 1), \
                 'mz_min': mz_min, 'mz_max': mz_max, \
                 'rt_min': rt_min, 'rt_max': rt_max, }
        fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor * ysm_max, 'I_smooth': ysm * ysm_max, 'I_raw': y * y_max,
                 'I_bl': y_bl * ysm_max, }
        od = {'flev': 2, 'crit': crit, 'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
        return od

    # general feature descriptors
    mz_maxI = mz[idx_maxI + 1]
    mz_mean = np.mean(mz)

    rt_maxI = x[idx_maxI + 1]
    rt_mean = np.mean(xsm)

    # create subset with equal endings
    _, icent = wImean(xsm, ybcor)

    # tailing
    tail = rt_maxI / rt_mean




    # d = np.min([icent, np.abs(icent - len(idxsm))])
    # ra=range((icent-d), (icent+d))

    # xr=xsm[ra]
    # yr=ybcor[ra]
    # yr_max = np.max(yr)
    # yr =  yr / yr_max
    a = integrate.trapz(ybcor*bcor_max, x=xsm)
    a_raw = integrate.trapz(y, x)
    a_sm = integrate.trapz(ysm, xsm)

    # I mind not need this here
    # include symmetry and monotonicity
    pl = find_peaks(ybcor*bcor_max, distance=10)
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

    qc = {'sId_gap': sid_gap, 'non_neg': nneg, 'sino': sino, 'raggedness': sdxdy, 'tail': tail, 'ppm': ppm}
    quant ={'raw': a_raw, 'sm': a_sm, 'smbl': a}
    descr = {'ndp': iLen, 'npeaks': plle, 'pprom': aa, 'st_span': np.round(rt_max - rt_min, 1),\
             'mz_maxI': mz_maxI, 'rt_maxI': rt_maxI, \
             'mz_mean': mz_mean, 'mz_min': mz_min, 'mz_max': mz_max, \
             'rt_mean': rt_mean,  'rt_min': rt_min, 'rt_max': rt_max,}
    fdata = {'st': x, 'st_sm': xsm, 'I_sm_bline': ybcor*ysm_max, 'I_smooth': ysm * ysm_max, 'I_raw': y * y_max, 'I_bl': y_bl * ysm_max,}
    od = {'flev': 3,  'qc': qc, 'quant': quant, 'descr': descr, 'fdata': fdata}
    return od



# def feat_summary(self, st_len_min=10, plot=True, mz_bound=0, rt_bound=0):
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Rectangle
#     import numpy as np
#     # import pandas as pd
#
#     self.feat = {}
#     self.feat_l2 = []
#     #idx = np.where(self.Xf[:,6] > 0)[0]
#     # filter for minimal length
#
#     min_le_bool = (self.cl_labels[1] >= st_len_min) & (self.cl_labels[0] != -1)
#     cl_n_above = self.cl_labels[0][min_le_bool]
#     self.feat.update(
#         {'id:' + str(x): {'flev': 1, 'crit': 'n < st_len_min'} for x in self.cl_labels[0][~min_le_bool]})
#
#     if plot:
#         cm = plt.cm.get_cmap('gist_ncar')
#         fig, ax = plt.subplots()
#         ax.set_facecolor('#EBFFFF')
#
#     # import time
#     # start = time.time()
#     # cc=0
#     for i in cl_n_above:
#         f1 = self.cl_summary(i)
#         ufid = 'id:' + str(i)
#         self.feat.update({ufid: f1})
#         if f1['flev'] == 2:
#             self.feat_l2.append(ufid)
#             if plot:
#                 ax.add_patch(Rectangle(((f1['rt_min'] - rt_bound), f1['mz_min'] - mz_bound), \
#                                        ((f1['rt_max']) - (f1['rt_min'])) + rt_bound * 2, \
#                                        (f1['mz_max'] - f1['mz_min']) + mz_bound * 2, fc='#C2D0D6',\
#                                        linestyle="solid", color='red', linewidth=2, zorder=0))
#
#     # stop = time.time()
#     # print(stop - start)
#
#     if plot:
#         idx = np.where(self.Xf[:, 4] <= 0)[0]  # noise
#         ax.scatter(self.Xf[idx, 3], self.Xf[idx, 1], s=0.2, c='gray', alpha=1)
#         idx = np.where(np.isin(self.Xf[:, 6], cl_n_above))[0]
#         im = ax.scatter(self.Xf[idx,3], self.Xf[idx, 1], c=np.log(self.Xf[idx,2]), s=5, cmap=cm)
#         fig.colorbar(im, ax=ax)
#         fig.show()
#
#     if len(self.feat_l2) == 0:
#         raise ValueError('No L2 features found')
#
#     #self.ftab = pd.DataFrame(feat).sort_values('ppm')


def feat_summary1(self):
    import time

    t1 = time.time()
    self.feat = {}
    self.feat_l2 = []
    self.feat_l3 = []
    print(f'Number of L0 features: {len(self.cl_labels[1])}')
    # filter for feature length
    min_le_bool = (self.cl_labels[1] >= self.qc_par['st_len_min']) & (self.cl_labels[0] != -1)
    cl_n_above = self.cl_labels[0][min_le_bool]
    print(f'Number of L1 features: {len(cl_n_above)}')

    self.feat.update({'id:' + str(x): {'flev': 1, 'crit': 'n < st_len_min'} for x in self.cl_labels[0][~min_le_bool]})
    t2 = time.time()
    # print('boiler')
    # print(t2-t1)

    c = 0
    for fid in cl_n_above:
        # print(fid)
        # print(f'{c}: {fid}')
        c +=1
        f1 = self.cl_summary(fid)
        #plt.plot(f1['fdata']['st'], f1['fdata']['I_raw'])
        ufid = 'id:' + str(fid)
        self.feat.update({ufid: f1})
        if f1['flev'] == 2:
            self.feat_l2.append(ufid)
        if f1['flev'] == 3:
            self.feat_l3.append(ufid)
    t3 = time.time()

    # print(f'run time per feature')
    # print((t3-t2)/len(cl_n_above))

    qq = [self.feat[x]['qc'][0] if isinstance(self.feat[x]['qc'], type(())) else self.feat[x]['qc'] for x in self.feat_l2 ]
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # out=pd.DataFrame(qq)
    # plt.scatter(out.sId_gap, out.non_neg, c=out.sino)


    if len(self.feat_l2) == 0:
        raise ValueError('No L2 features found')
    else:
        print(f'Number of L2 features: {len(self.feat_l2)}')

    if len(self.feat_l3) == 0:
        raise ValueError('No L3 features found')
    else:
        print(f'Number of L3 features: {len(self.feat_l3)}')


def peak_picking(self, st_adj=6e2, q_noise=0.89, dbs_par={'eps': 0.02, 'min_samples': 2},
                 selection={'mz_min': None, 'mz_max': None, 'rt_min': None, 'rt_max': None},
                 qc_par = {'st_len_min': 5, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 2, 'ppm': 15}):
    import numpy as np
    from sklearn.cluster import DBSCAN
    import time

    start = time.time()
    self.st_adj = st_adj
    self.q_noise = q_noise
    self.dbs_par = dbs_par
    self.qc_par = qc_par
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
    s1 = time.time()
    # print(s1-start)
    #plt.scatter(X2[:,0], X2[:,1])
    #oo= self.vis_spectrum(q_noise=self.q_noise, selection=selection)
    #return(X2)
    # perform clustering
    # print('start cl')
    dbs = DBSCAN(eps=self.dbs_par['eps'], min_samples=self.dbs_par['min_samples'], algorithm='auto').fit(self.Xf[idx_signal, :][..., np.array([1, 5])])
    #self.cl_membership = dbs.labels_
    # print('cl done')
    s2=time.time()
    # print(s2-s1)

    # append cluster membership to windowed Xraw
    self.cl_labels = np.unique(dbs.labels_, return_counts=True)
    # print('cl unique done')

    cl_mem = np.ones(self.Xf.shape[0]) * -1
    cl_mem[idx_signal] = dbs.labels_
    self.Xf = np.c_[self.Xf, cl_mem]    # Xf has 7 columns: scanId, mz, int, st, noiseBool, st_adj, clMem
    # print('cl append done')
    s3=time.time()
    # print(s3-s2)

    # qc filtering
    # self.feat_summary(st_len_min=10, plot=plot, mz_bound=0, rt_bound=0)  # self.feat: list
    s4 = time.time()
    # print('feat filter')
    self.feat_summary1()
    s5 = time.time()

    # # append clMem to X
    #
    # self.cl_labels = np.unique(dbs.labels_, return_counts=True)
    # print('Number of L1 features: ' + str(len(self.cl_labels[0])))
    #
    # # feature description and QC filtering
    # self.feat_summary(st_len_min=10, plot=plot, mz_bound=0, rt_bound=0) # self.feat: list
    # print('Number of L2 features: ' + str(len(self.feat_l2)))

    # plot here if requested
    #self.fs = est_intensities(self.ftab, self.cl_membership, self.Xf)
    # return fs


def vis_feature_pp(self, selection={'mz_min': 425, 'mz_max': 440, 'rt_min': 400, 'rt_max': 440}, lev=3):
        # l2 and l1 features mz/rt plot after peak picking
        # label feature with id
        # two panel plot: bottom m/z over st, top: st vs intentsity upon click on feature
        # basically like vis spectrum but including rectangles and all points but those in fgr feature(s) are s=0.1 and c='grey
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        from matplotlib.backend_bases import MouseButton
        from matplotlib.patches import Rectangle
        import numpy as np
        import pandas as pd

        # data = np.array([list(qc.keys()), list(qc.values())]).T
        # # Add a table at the bottom of the axes
        # plt.table(cellText=list(qc))
        #
        # fig, ax = plt.subplots(1, 1)
        # column_labels = ["da", "a"]
        # ax.axis('tight')
        # ax.axis('off')
        # ax.table(cellText=data, colLabels=column_labels, loc="center")

        def vis_feature(fdict, id, ax=None, add=False):
            import matplotlib.pyplot as plt

            # idx = np.where(self.Xf[:, 6] == fdict['id'])[0]
            if isinstance(ax, type(None)):
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(fdict['fdata']['st'], fdict['fdata']['I_raw'], label='raw', linewidth=0.5, color='black', zorder=0)

            if fdict['flev'] == 3:
                ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_smooth'], label='smoothed', linewidth=0.5, color='cyan', zorder=0)
                ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_bl'], label='baseline', linewidth=0.5, color='black', ls='dashed', zorder=0)
                ax.plot(fdict['fdata']['st_sm'], fdict['fdata']['I_sm_bline'], label='smoothed bl-cor', color='orange', zorder=0)
            if not add:
                ax.set_title(id + f',  flev: {fdict["flev"]}')
                ax.legend()
            else:
                old= axs[0].get_title()
                ax.set_title(old + '\n' +  id + f',  flev: {fdict["flev"]}')
            ax.set_ylabel(r"$\bfCount$")
            ax.set_xlabel(r"$\bfScan time$, s")

            data = np.array([list(fdict['qc'].keys()), np.round(list(fdict['qc'].values()), 2)]).T
            column_labels = ["descr", "value"]
            ax.table( cellText=data, colLabels=column_labels, loc='lower right', colWidths = [0.1]*3)

            return ax

        # Xf has 7 columns: 0 - scanId, 1 - mz, 2 - int, 3 - st, 4 - noiseBool, 5 - st_adj, 6 - clMem

        # which feature - define plot axis boundaries
        Xsub = self.window_mz_rt(self.Xf, selection, allow_none=False, return_idc=False)
        fid_window = np.unique(Xsub[:,6])
        from itertools import compress
        l2_ffeat = list(compress(self.feat_l2, np.in1d([float(i.split(':')[1]) for i in self.feat_l2], fid_window)))
        l3_ffeat = list(compress(self.feat_l3, np.in1d([float(i.split(':')[1]) for i in self.feat_l3], fid_window)))



        cm = plt.cm.get_cmap('gist_ncar')
        fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8,10), constrained_layout=False)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axs[0].set_ylabel(r"$\bfCount$")
        axs[0].set_xlabel(r"$\bfScan time$, s")

        vis_feature(self.feat[self.feat_l2[0]], id=self.feat_l2[0], ax=axs[0])
        axs[1].set_facecolor('#EBFFFF')
        axs[1].text(1.03, 0, self.df.fname[0], rotation=90, fontsize=6, transform=axs[1].transAxes)
        axs[1].scatter(Xsub[Xsub[:,4]==0, 3], Xsub[Xsub[:,4]==0, 1], s=0.1, c='gray', alpha=0.5)


        # coords = []

        # coords=np.array([[self.feat[x]['rt_min'], self.feat[x]['rt_max'], self.feat[x]['mz_min'], self.feat[x]['mz_max']] for x in self.feat_l2])
        # ids = [str(i.split(':')[1]) for i in self.feat_l2]
        # axs[1].text(coords[:,1], coords[:,3], np.array(ids),
        #             bbox=dict(facecolor='red', alpha=0.3, boxstyle='circle', edgecolor='white'), picker=True)

        if lev < 3:
            for i in l2_ffeat :
                fid = float(i.split(':')[1])
                # Xsub[Xsub[:,6] == fid, :]
                f1 = self.feat[i]

                if(f1['flev'] == 2):
                    a_col= 'red'
                    fsize = 6
                    mz=Xsub[Xsub[:, 6] == fid, 1]
                    if len(mz) == len(f1['fdata']['st']):
                        im = axs[1].scatter(f1['fdata']['st'], mz,
                                            c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
                        axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                                   ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                                   (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                                   linestyle="solid", color='red', linewidth=2, zorder=0, in_layout=True,
                                                   picker=False))
                        axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                        bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                                  in_layout=False),
                                        wrap=False, picker=True, fontsize=fsize)

        if lev == 3:
            for i in l3_ffeat:
                fid = float(i.split(':')[1])
                f1 = self.feat[i]

                if (f1['flev'] == 3):
                    a_col = 'green'
                    fsize = 10
                    mz = Xsub[Xsub[:, 6] == fid, 1]
                    if len(mz) == len(f1['fdata']['st']):
                        axs[1].add_patch(Rectangle(((f1['descr']['rt_min']), f1['descr']['mz_min']), \
                                                   ((f1['descr']['rt_max']) - (f1['descr']['rt_min'])), \
                                                   (f1['descr']['mz_max'] - f1['descr']['mz_min']), fc='#C2D0D6', \
                                                   linestyle="solid", color='red', linewidth=2, zorder=0, in_layout=True,
                                                   picker=False))
                        axs[1].annotate(i.split(':')[1], (f1['descr']['rt_max'], f1['descr']['mz_max']),
                                        bbox=dict(facecolor=a_col, alpha=0.3, boxstyle='circle', edgecolor='white',
                                                  in_layout=False),
                                        wrap=False, picker=True, fontsize=fsize)
                        im = axs[1].scatter(f1['fdata']['st'], mz,
                                            c=np.log(f1['fdata']['I_raw']), s=5, cmap=cm)
        # coords = pd.DataFrame(coords)
        cbaxes = inset_axes(axs[1], width="30%", height="5%", loc=3)
        plt.colorbar(im, cax=cbaxes, ticks=[0., 1], orientation='horizontal')

        def p_text(event):
            # print(event.artist)
            ids = str(event.artist.get_text())
            if event.mouseevent.button is MouseButton.LEFT:
                # print('left click')
                # print('id:' + str(ids))
                axs[0].clear()
                axs[0].set_title('')
                vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=False)
                event.canvas.draw()
            if event.mouseevent.button is MouseButton.RIGHT:
                # print('right click')
                # print('id:' + str(ids))
                vis_feature(self.feat['id:' + ids], id=ids, ax=axs[0], add=True)
                event.canvas.draw()

        cid1 = fig.canvas.mpl_connect('pick_event', p_text)
        axs[1].set_xlabel(r"$\bfScan time$, s")
        axs[1].yaxis.offsetText.set_visible(False)
        axs[1].yaxis.set_label_text(r"$\bfm/z$")
#
#
# test=pd.DataFrame([self.feat[x]['qc'] for x in self.feat_l2] + [self.feat[x]['qc'] for x in self.feat_l3])
# test['id'] = self.feat_l2 + self.feat_l3
#
# test=test.dropna(subset=['sino'])
# plt.scatter([self.feat[x]['descr']['mz_maxI'] for x in test.sort_values(by=['sino'], ascending=False).id[0:30]],
# [self.feat[x]['descr']['rt_maxI'] for x in test.sort_values(by=['sino'], ascending=False).id[0:30]])
#
# plt.scatter(np.log(test.sino), test.ppm)