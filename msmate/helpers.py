# msmate, v2.0.1
# author: T Kimhofer
# Murdoch University, July 2022

# dependency import
import base64
import numpy as np
import zlib
import re
import obonet
# from tqdm import tqdm
# import time
# import pandas as pd
# import subprocess
# import xml.etree.ElementTree as ET


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
        iis, ft = _read_bin(s, obos)
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
        d = np.frombuffer(zlib.decompress(base64.b64decode(dbin.text)), dtype=dt).tolist()
    else:
        d = np.frombuffer(base64.b64decode(dbin.text), dtype=dt).tolist()
    # return data and meta
    out = {'i': [{x: obo_ids[x]['name']} for x in dvars.keys() if x in obo_ids.keys()], 'd': d}

    return (out, ft)

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