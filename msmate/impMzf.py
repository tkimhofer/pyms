
# msmate, v2.0.1
# author: T Kimhofer
# Murdoch University, July 2022

# dependency import
import base64
import numpy as np
import zlib
import re
import obonet
from tqdm import tqdm
import time
import pandas as pd
import subprocess
import xml.etree.ElementTree as ET

def rec_attrib(s, ii={}):
    # recursively collecting cvParam
    for at in list(s):
        if 'cvParam' in at.tag:
            if 'name' in at.attrib.keys():
                ii.update({at.attrib['name'].replace(' ', '_'): at.attrib['value']})
        if len(list(at)) > 0:
            rec_attrib(at, ii)

    return ii

def extract_mass_spectrum_lev1(i, s_level, s_dic):
    s = s_level[i]
    iistart = s_dic[i]
    ii = rec_attrib(s, iistart)
    # Extract data
    # two binary arrays: m/z value and intensity
    # read m/z value
    le = len(list(s)) - 1
    # pars = list(list(list(s)[7])[0]) # ANPC files
    pars = list(list(list(s)[le])[0])  # NPC files

    dt = arraydtype(pars)

    # check if compression
    if pars[1].attrib['name'] == 'zlib compression':
        mz = np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
    else:
        mz = np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)

    # read intensity values
    pars = list(list(list(s)[le])[1])
    dt = arraydtype(pars)
    if pars[1].attrib['name'] == 'zlib compression':
        I = np.frombuffer(zlib.decompress(base64.b64decode(pars[3].text)), dtype=dt)
    else:
        I = np.frombuffer(base64.b64decode(pars[3].text), dtype=dt)

    return (ii, mz, I)

def _collect_spectra_chrom(s, ii={}, d=9, c=0, flag='1', tag='', obos={}):
    if c == d: return
    if (('chromatogram' == tag) | ('spectrum' == tag)):
        rtype = 'Ukwn'
        if ('chromatogram' == tag):
            rtype = 'c'
        if ('spectrum' == tag):
            rtype = 's'
        add_meta, data = _extr_spectrum_chromg(s, flag=flag, rtype=rtype, obos=obos)
        # if rtype == 'c': print(tag); print(add_meta); print(data)
        ii.update({rtype + add_meta['index']: {'meta': add_meta, 'data': data}})

    ss = list(s)
    if len(ss) > 0:
        for i in range(len(ss)):
            tag = re.sub('\{.*\}', '', ss[i].tag)
            _collect_spectra_chrom(ss[i], ii=ii, d=d, c=c + 1, flag=flag, tag=tag, obos=obos)

    return ii

def _extr_spectrum_chromg(x, flag='1', rtype='s', obos=[]):
    add_meta = _vaPar_recurse(x, ii={}, d=6, c=0, dname='name', dval='value', ddt='all')

    if rtype == 's':  # spectrum
        if flag == '1':  # mslevel 1 input
            if ('MS:1000511' in add_meta.keys()):  # find mslevel information
                if add_meta['MS:1000511'] == '1':  # read in ms1
                    out = _data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
                else:
                    return (add_meta, [])
            else:  # can;t find mslevel info
                print('MS level information not found not found - reading all experiments.')
                out = _data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
        else:
            out = _data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
    else:
        out = _data_recurse1(x, ii={}, d=9, c=0, ip='', obos=obos)
    return (add_meta, out)

def _data_recurse1(s, ii={}, d=9, c=0, ip='', obos=[]):
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    if ('binaryDataArray' == re.sub('\{.*\}', '', s.tag)):
        iis, ft = read_bin(s, obos)
        ii.update({ip + ft: iis})
    # if ('cvParam' in s.tag):
    ss = list(s)
    if len(ss) > 0:
        for i in range(len(ss)):
            tag = re.sub('\{.*\}', '', ss[i].tag)
            if ('chromatogram' == tag):
                ip = 'c'
            if ('spectrum' == tag):
                ip = 's'
            _data_recurse1(s=ss[i], ii=ii, d=d, c=c + 1, ip=ip, obos=obos)
    return ii

def _dt_co(dvars):
    # dtype and compression
    dt = None  # data type
    co = None  # compression
    ft = 'ukwn'  # feature type
    if 'MS:1000523' in dvars.keys():
        dt = np.dtype('<d')

    if 'MS:1000522' in dvars.keys():
        dt = np.dtype('<i8')

    if 'MS:1000521' in dvars.keys():
        dt = np.dtype('<f')

    if 'MS:1000519' in dvars.keys():
        dt = np.dtype('<i4')

    if isinstance(dt, type(None)):
        raise ValueError('Unknown variable type')

    if 'MS:1000574' in dvars.keys():
        co = 'zlib'

    if 'MS:1000514' in dvars.keys():
        ft = 'm/z'
    if 'MS:1000515' in dvars.keys():
        ft = 'Int'

    if 'MS:1000595' in dvars.keys():
        ft = 'time'
    if 'MS:1000516' in dvars.keys():
        ft = 'Charge'
    if 'MS:1000517' in dvars.keys():
        ft = 'sino'

    return (dt, co, ft)

def _read_bin(k, obo_ids):
    # k is binary array
    # collect metadata
    dvars = _vaPar_recurse(k, ii={}, d=3, c=0, dname='accession', dval='cvRef', ddt='-')
    dt, co, ft = _dt_co(dvars)

    child = _children(k)
    dbin = k[child.index('binary')]

    if co == 'zlib':
        d = np.frombuffer(zlib.decompress(base64.b64decode(dbin.text)), dtype=dt)
    else:
        d = np.frombuffer(base64.b64decode(dbin.text), dtype=dt)
    # return data and meta
    out = {'i': [{x: obo_ids[x]['name']} for x in dvars.keys() if x in obo_ids.keys()], 'd': d}

    return (out, ft)

def _vaPar_recurse(s, ii={}, d=9, c=0, dname='accession', dval='value', ddt='all'):
    import re
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    if ('cvParam' in s.tag):
        iis = {}
        name = s.attrib[dname]
        value = s.attrib[dval]
        if value == '': value = True
        iis.update({name: value})
        if 'unitCvRef' in s.attrib:
            iis.update({s.attrib['unitAccession']: s.attrib['unitName']})
        ii.update(iis)
    else:
        if ddt == 'all':
            iis = s.attrib
            ii.update(iis)
    # if ('cvParam' in s.tag):
    ss = list(s)
    if len(ss) > 0:
        # if ('spectrum' in s.tag):
        for i in range(len(ss)):
            _vaPar_recurse(s=ss[i], ii=ii, d=d, c=c + 1, ddt=ddt)
    return ii

def _get_obo(obos, obo_ids={}):
    # download obo annotation data
    # create single dict with keys being of obo ids, eg, MS:1000500
    # obos is first cv element in mzml node
    for o in obos:
        id=o.attrib['id']
        if id not in obo_ids.keys():
            gr = obonet.read_obo(o.attrib['URI'])
            gv = gr.nodes(data=True)
            map = {id_: {'name': data.get('name'), 'def': data.get('def'), 'is_a': data.get('is_A')} for
                          id_, data in gv}
            #obo_ids.update({id: {'inf': o.attrib, 'data': map}})
            obo_ids.update(map)

    return obo_ids

def _children(xr):
    return [re.sub('\{.*\}', '', x.tag) for x in list(xr)]

def exp_meta(root, ind, d):
    # combine attributes to dataframe with column names prefixed acc to child/parent relationship
    # remove web links in tags
    filed = _node_attr_recurse(s=root, d=d, c=0, ii=[])

    dn = {}
    for i in range(len(filed)):
        dn.update(dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
                           list(filed[i].values())[1:])))
    return pd.DataFrame(dn, index=[ind])

def _node_attr_recurse(s, ii=[], d=3, c=0,  pre=0):
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    # define ms level
    if ('list' not in s.tag) | ('spectrum' not in s.tag):
        iis = {}
        iis['path'] = re.sub('\{.*\}', '', s.tag) + '_' + str(pre)
        iis.update(s.attrib)
        ii.append(iis)
    if len(list(s)) > 0:
        if ('spectrum' not in s.tag.lower()):
            ss = list(s)
            for i in range(len(list(ss))):
                at = ss[i]
                _node_attr_recurse(s=at, ii=ii,d=d, c=c + 1, pre=i)
    return ii

def read_exp(path, ftype='mzML', plot_chrom=False, n_max=1000, only_summary=False, d=4):
    cmd = 'find ' + path + ' -type f -iname *'+ ftype
    sp = subprocess.getoutput(cmd)
    files = sp.split('\n')

    if len(files) == 0:
        raise ValueError(r'No .' + ftype + ' files found in path: '+path)

    if len(files) > n_max:
        files=files[0:n_max]

    if only_summary:
        print('Creating experiment summary.')
        #import hashlib
        #hashlib.md5(open(fid,'rb').read()).hexdigest()

        df_summary = []
        for i in range(len(files)):
            print(files[i])
            tree = ET.parse(files[i])
            root = tree.getroot()
            #tt = exp_meta(root=root, ind=i, d=d).T
            df_summary.append(exp_meta(root=root, ind=i, d=d))

        df = pd.concat(df_summary).T
        return df
    else:
        data=[]
        df_summary = []
        print('Importing '+ str(len(files)) + ' files.')
        for i in range(len(files)):
            tree = ET.parse(files[i])
            root = tree.getroot()
            df_summary.append(exp_meta(root=root, ind=i, d=d))
            out = read_mzml_file(root, ms_lev=1, plot=plot_chrom)
            data.append(out[0:3])

        df = pd.concat(df_summary).T
        return (df, data)

def read_mzml_file(root, ms_lev=1, plot=True):
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

    # if plot:
    #     pl = plt_chromatograms(df, tmin, tmax)
    #     return (mz, I, df, pl)
    # else:
    return (mz, I, df)

def arraydtype(pars):
    # determines base64 single or double precision, all in little endian
    pdic = {}
    for pp in pars:
        if 'name' in pp.attrib.keys():
            pdic.update({pp.attrib['accession'].replace(' ', '_'): pp.attrib['name']})
        else:
            pdic.update(pp.attrib)

    dt = None
    if 'MS:1000523' in pdic.keys():
        dt = np.dtype('<d')

    if 'MS:1000522' in pdic.keys():
        dt = np.dtype('<i8')

    if 'MS:1000521' in pdic.keys():
        dt = np.dtype('<f')

    if 'MS:1000519' in pdic.keys():
        dt = np.dtype('<i4')

    if isinstance(dt, type(None)):
        raise ValueError('Unknown variable type')

    return dt
