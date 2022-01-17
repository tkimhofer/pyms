



def read_exp(path, ftype='mzML', plot_chrom=False, n_max=1000):
    import subprocess
    import xml.etree.ElementTree as ET
    # import pandas as pd
    # import numpy as np

    cmd = 'find ' + path + ' -type f -iname *'+ ftype
    sp = subprocess.getoutput(cmd)
    files = sp.split('\n')

    if len(files) > n_max:
        files=files[0:n_max]

    data=[]
    for fid in files:
        print(fid)
        tree = ET.parse(fid)
        root = tree.getroot()
        out = read_mzml_file(root, ms_lev=1, plot=plot_chrom)
        data.append(out[0:3])

    return data


def read_mzml_file(root, ms_lev=1, plot=True):
    from tqdm import tqdm
    import time
    import numpy as np
    import pandas as pd
    # file specific variables
    le = len(list(root[0]))-1
    id = root[0][le].attrib['id']


    StartTime=root[0][le].attrib['startTimeStamp'] # file specific
    #ii=root[0][6][0].attrib

    # filter ms level spectra
    s_all = list(root[0][le][0])

    s_level=[]
    s_dic=[]
    for s in s_all:
        scan = s.attrib['id'].replace('scan=', '')
        arrlen = s.attrib['defaultArrayLength']

        ii = {'fname': id, 'acquTime': StartTime, 'scan_id': scan, 'defaultArrayLength': arrlen}
        if 'name' in s[0].attrib.keys():
            if ((s[0].attrib['name'] == 'ms level') & (s[0].attrib['value'] == str(ms_lev))) | (s[0].attrib['name'] == 'MS1 spectrum'):
                # desired ms level
                s_level.append(s)
                s_dic.append(ii)
    print('Number of MS level '+ str(ms_lev)+ ' mass spectra : ' + str(len(s_level)))
    # loop over spectra
    # # check multicore
    # import multiprocessing
    # from joblib import Parallel, delayed
    # from tqdm import tqdm
    # import time
    # ncore = multiprocessing.cpu_count() - 2
    # #spec_seq = tqdm(np.arange(len(s_level)))
    # spec_seq = tqdm(np.arange(len(s_level)))
    # a= time.time()
    # bls = Parallel(n_jobs=ncore, require='sharedmem')(delayed(extract_mass_spectrum_lev1)(i, s_level, s_dic) for i in spec_seq)
    # b=time.time()
    # extract mz, I ,df
    # s1=(b-a)/60

    spec_seq = np.arange(len(s_level))
    #spec_seq = np.arange(10)
    ss = []
    I = []
    mz = []
    a = time.time()
    for i in tqdm(spec_seq):
        ds, ms, Is = extract_mass_spectrum_lev1(i, s_level, s_dic)
        ss.append(ds)
        mz.append(ms)
        I.append(Is)

    b = time.time()
    s2 = (b - a) / 60

    df = pd.DataFrame(ss)

    swll = df['scan_window_lower_limit'].unique()
    swul = df['scan_window_upper_limit'].unique()

    if len(swll) ==1 & len(swul) == 1:
        print('M/z scan window: ', swll[0] + "-" + swul[0] + ' m/z')
    else:
        print('Incosistent m/z scan window!')
        print('Lower bound: '+str(swll)+' m/z')
        print('Upper bound: '+str(swul)+' m/z')


    #df.groupby('scan_id')['scan_start_time'].min()

    df.scan_start_time=df.scan_start_time.astype(float)

    tmin=df.scan_start_time.min()
    tmax=df.scan_start_time.max()
    print('Recording time: ' + str(np.round(tmin,1)) + '-' + str(np.round(tmax,1)), 's (' + str(np.round((tmax-tmin)/60, 1)), 'm)')
    print('Read-in time: ' + str(np.round(s2, 1)) + ' min')
    # df.scan_id = df.scan_id.astype(int)
    df.scan_id = np.arange(1, df.shape[0]+1)
    df.total_ion_current=df.total_ion_current.astype(float)
    df.base_peak_intensity=df.base_peak_intensity.astype(float)

    if plot:
        pl = plt_chromatograms(df, tmin, tmax)
        return (mz, I, df, pl)
    else:
        return (mz, I, df)

def plt_chromatograms(df, tmin, tmax):
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax1.set_title(df.fname[0], fontsize=8, y=-0.15)
    ax1.text(1.04, 0, df.fname[0],  rotation=90, fontsize=6, transform=ax1.transAxes)
    ax1.plot(df.scan_start_time, df.total_ion_current, label='TIC')
    ax1.plot(df.scan_start_time, df.base_peak_intensity, label='BPC')
    fig.canvas.draw()
    offset = ax1.yaxis.get_major_formatter().get_offset()
    # offset = ax1.get_xaxis().get_offset_text()

    ax1.set_xlabel(r"$\bfs$")
    ax2 = ax1.twiny()

    ax1.yaxis.offsetText.set_visible(False)
    # ax1.yaxis.set_label_text("original label" + " " + offset)
    ax1.yaxis.set_label_text(r"$\bfInt/count$ ($x 10^" + offset.replace('1e', '') + '$)')

    def tick_conv(X):
        V = X / 60
        return ["%.2f" % z for z in V]

    tick_loc = np.linspace(0, np.round((tmax - tmin) / 60, 0), (round(np.round((tmax - tmin) / 60, 0) / 2) + 1)) * 60
    # tick_loc = np.arange(np.round((tmax - tmin) / 60, 0), 0.1)*60
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(tick_loc)
    ax2.set_xticklabels(tick_conv(tick_loc))
    ax2.set_xlabel(r"$\bfmin$")
    # ax2.yaxis.set_label_text(r"$Int/count$ " + offset)

    ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))

    fig.show()
    return (fig, ax1, ax2)

def rec_attrib(s, ii={}):
    # recursively collecting cvParam
    for at in list(s):
        if 'cvParam' in at.tag:
            if 'name' in at.attrib.keys():
                ii.update({at.attrib['name'].replace(' ', '_'): at.attrib['value']})
        if len(list(at)) >0:
            rec_attrib(at, ii)

    return ii

def extract_mass_spectrum_lev1(i, s_level, s_dic):
        import base64
        import numpy as np
        import zlib

        s=s_level[i]
        iistart = s_dic[i]
        ii =rec_attrib(s, iistart)
        # Extract data
        # two binary arrays: m/z value and intensity
        # read m/z value
        le = len(list(s))-1
        #pars = list(list(list(s)[7])[0]) # ANPC files
        pars = list(list(list(s)[le])[0]) # NPC files

        dt = arraydtype(pars)

        # check if compression
        if pars[1].attrib['name'] == 'zlib compression':
            mz= np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
        else:
             mz=np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)

        # read intensity values
        pars = list(list(list(s)[le])[1])
        dt = arraydtype(pars)
        if pars[1].attrib['name'] == 'zlib compression':
            I = np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
        else:
            I=np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)

        return (ii, mz, I)

def arraydtype(pars):
    import numpy as np
    # determines base64 single or double precision, all in little endian
    pdic = {}
    for pp in pars:
        if 'name' in pp.attrib.keys():
            pdic.update({pp.attrib['accession'].replace(' ', '_'): pp.attrib['name']})
        else:
            pdic.update(pp.attrib)

    if 'MS:1000576' in pdic.keys():
        if pdic['MS:1000576'] != 'no compression':
            raise ValueError('Decompression not implemented yet')

    dt = None
    if 'MS:1000523' in pdic.keys():
        # mz array data
        if pdic['MS:1000523'] == '64-bit float':
            dt = np.dtype('<d')
        if pdic['MS:1000523'] == '32-bit float':
            dt = np.dtype('<f')
    if 'MS:1000521' in pdic.keys():
        # mz array data
        if pdic['MS:1000521'] == '64-bit float':
            dt = np.dtype('<d')
        if pdic['MS:1000521'] == '32-bit float':
            dt = np.dtype('<f')

    if isinstance(dt, type(None)):
        raise ValueError('Unknown variable type')

    return dt

