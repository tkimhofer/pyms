import base64
import re
import numpy as np
import zlib
import obonet
import xml.etree.ElementTree as ET
import isodate
from collections import Counter
from pathlib import Path

def _get_scan_time_seconds_from_node(x):
    """
    Extract mzML scan start time from spectrum XML node.
    Returns:
        (rt_sec, time_unit_ori)
    """

    for elem in x.iter():
        tag = re.sub(r"\{.*\}", "", elem.tag)

        if tag != "cvParam":
            continue

        if elem.attrib.get("accession") == "MS:1000016":
            rt = float(elem.attrib["value"])

            unit_acc = elem.attrib.get("unitAccession")
            unit_name = elem.attrib.get("unitName", "").lower()

            if unit_acc == "UO:0000010" or unit_name in {"second", "seconds", "sec", "s"}:
                return rt, "second"

            if unit_acc == "UO:0000031" or unit_name in {"minute", "minutes", "min"}:
                return rt * 60.0, "minute"

            raise ValueError(
                f"Unknown scan time unit for MS:1000016: "
                f"unitAccession={unit_acc}, unitName={unit_name}"
            )

    return None, None


def _collect_spectra_chrom(s, ii=None, d=9, c=0, flag:str='1', tag='', obos=None, rt_min=None, rt_max=None):
    # this is recursion
    if ii is None: ii = {}
    if obos is None: obos = {}

    if c == d: return ii

    if tag in {"chromatogram", "spectrum"}:
        rtype = 'Ukwn'
        if ('chromatogram' == tag):
            rtype = 'c'
        if ('spectrum' == tag):
            rtype = 's'
        add_meta, data = _extr_spectrum_chromg(s, flag=flag, rtype=rtype, obos=obos, rt_min=rt_min, rt_max=rt_max)
        # if rtype == 'c': print(tag); print(add_meta); print(data)
        if data != []:
            ii.update({rtype + add_meta['index']: {'meta': add_meta, 'data': data}})

    for child in list(s):
        tag = re.sub('\{.*\}', '', child.tag)
        _collect_spectra_chrom(child, ii=ii, d=d, c=c + 1, flag=flag, tag=tag, obos=obos, rt_min=rt_min, rt_max=rt_max)

    return ii


def _data_recurse1(s, ii=None, d=9, c=0, ip='', obos=None):
    # recursively extracts attributes from node s and with depth  d,
    # add label children iterator as prefix
    if c == d: return ii
    if ii is None: ii = {}
    if obos is None: obos = []
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
    dvars = _vaPar_recurse(k, d=3, c=0, dname='accession', dval='cvRef', ddt='-')
    dt, co, ft = _dt_co(dvars)
    child = _children(k)
    dbin = k[child.index('binary')]
    if co == 'zlib':
        d = np.frombuffer(zlib.decompress(base64.b64decode(dbin.text)), dtype=dt)#.tolist()
    else:
        d = np.frombuffer(base64.b64decode(dbin.text), dtype=dt)#.tolist()
    # return data and meta
    out = {'i': [{x: obo_ids[x]['name']} for x in dvars.keys() if x in obo_ids.keys()], 'd': d}

    return (out, ft)


def _get_obo(obos, obo_ids=None):
    # download obo annotation data... this is too fragile (old sourceforge links break this code)
    # providing stable web-links for main ontologies PSI-MS (Human Proteome Organisation on Github) and
    # Unit Ontology (UO) from OBO Foundry

    # DEFAULT_PATH = {
    #     'MS': 'https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo',
    #     'UO': 'http://purl.obolibrary.org/obo/uo.obo'
    # }

    if obo_ids is None: obo_ids={}

    DEFAULT_PATH = {
        'MS': 'msmate/ontologies/psi-ms.obo',
        'UO': 'msmate/ontologies/uo.obo'
    }

    # create single dict with keys being of obo ids, eg, MS:1000500
    # obos is first cv element in mzml node
    for o in obos:
        cv_id = o.attrib['id']
        if cv_id in obo_ids:
            continue
        gr = None
        try:
            gr = obonet.read_obo(o.attrib['URI'])
        except Exception:
            if cv_id in DEFAULT_PATH:
                gr = obonet.read_obo(DEFAULT_PATH[cv_id])
            else:
                continue
        if gr is None:
            continue
        gv = gr.nodes(data=True)
        map_ = {
            node_id: {
                "name": data.get("name"),
                "def": data.get("def"),
                "is_a": data.get("is_a")
            }
            for node_id, data in gv
        }
        obo_ids.update(map_)
    return obo_ids


def _children(xr):
    return [re.sub('\{.*\}', '', x.tag) for x in list(xr)]


def _node_attr_recurse(s, ii=None, d=3, c=0,  pre=0):
    # recursively extracts attributes from node s and with depth d, add label children iterator as prefix
    if c == d: return
    if ii is None: ii = []
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

def _vaPar_recurse(s, ii=None, d=9, c=0, dname='accession', dval='value', ddt='all'):
    import re
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    if ii is None: ii = {}
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


def _collect_spectra_chrom(s, ii=None, d=9, c=0, flag='1', tag='', obos=None, rt_min=None, rt_max=None):
    # this is recursion
    if ii is None: ii = {}
    if obos is None: obos = {}

    if c == d: return ii

    if tag in {"chromatogram", "spectrum"}:
        rtype = 'Ukwn'
        if ('chromatogram' == tag):
            rtype = 'c'
        if ('spectrum' == tag):
            rtype = 's'
        add_meta, data = _extr_spectrum_chromg(s, flag=flag, rtype=rtype, obos=obos, rt_min=rt_min, rt_max=rt_max)
        # if rtype == 'c': print(tag); print(add_meta); print(data)
        if data != []:
            ii.update({rtype + add_meta['index']: {'meta': add_meta, 'data': data}})

    for child in list(s):
        tag = re.sub('\{.*\}', '', child.tag)
        _collect_spectra_chrom(child, ii=ii, d=d, c=c + 1, flag=flag, tag=tag, obos=obos, rt_min=rt_min, rt_max=rt_max)

    return ii


def _extr_spectrum_chromg(x, flag='1', rtype='s', obos=None, rt_min=None, rt_max=None,):
    ### fct skips extracting data if scantime not in range defined in rt_min/rt_max (saving time)
    if obos is None: obos = {}

    add_meta = _vaPar_recurse(x, d=6, c=0, dname='name', dval='value', ddt='all')

    if rtype == "s": # spectrum
        rt_sec, time_unit_ori = _get_scan_time_seconds_from_node(x)

        if rt_sec is not None:
            add_meta["MS:1000016"] = rt_sec
            add_meta["time_unit"] = "sec"
            add_meta["time_unit_ori"] = time_unit_ori

            if rt_min is not None and rt_sec < rt_min:
                return add_meta, []

            if rt_max is not None and rt_sec > rt_max:
                return add_meta, []

        if flag == '1':  # mslevel 1 input
            if ('MS:1000511' in add_meta): # mslevel info
                if add_meta['MS:1000511'] != '1':  # do not read
                    return (add_meta, [])
            else:
                # no mslevel info -> could happen if experimental setup comprises non-MS / auxiliary detector modes (eg.UV)
                # skipping these
                # print(add_meta)
                # breakpoint()
                return (add_meta, [])
                # print('MS level information not found - reading all experiments.')

    out = _data_recurse1(x, d=9, c=0, ip='', obos=obos)
    return add_meta, out


def _data_recurse1(s, ii=None, d=9, c=0, ip='', obos=None):
    # recursively extracts attributes from node s and with depth  d,
    # add label children iterator as prefix
    if c == d: return ii
    if ii is None: ii = {}
    if obos is None: obos = {}
    if ('binaryDataArray' == re.sub('\{.*\}', '', s.tag)):
        iis, ft = _read_bin(s, obos)
        ii.update({ip + ft: iis})

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
    dvars = _vaPar_recurse(k, d=3, c=0, dname='accession', dval='cvRef', ddt='-')
    dt, co, ft = _dt_co(dvars)
    child = _children(k)
    dbin = k[child.index('binary')]
    if co == 'zlib':
        d = np.frombuffer(zlib.decompress(base64.b64decode(dbin.text)), dtype=dt)#.tolist()
    else:
        d = np.frombuffer(base64.b64decode(dbin.text), dtype=dt)#.tolist()
    # return data and meta
    out = {'i': [{x: obo_ids[x]['name']} for x in dvars.keys() if x in obo_ids.keys()], 'd': d}

    return (out, ft)


def _get_obo(obos, obo_ids=None):
    # download obo annotation data... this is too fragile (old sourceforge links break this code)
    # providing stable web-links for main ontologies PSI-MS (Human Proteome Organisation on Github) and
    # Unit Ontology (UO) from OBO Foundry

    # DEFAULT_PATH = {
    #     'MS': 'https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo',
    #     'UO': 'http://purl.obolibrary.org/obo/uo.obo'
    # }

    if obo_ids is None: obo_ids={}

    DEFAULT_PATH = {
        'MS': 'msmate/ontologies/psi-ms.obo',
        'UO': 'msmate/ontologies/uo.obo'
    }

    # create single dict with keys being of obo ids, eg, MS:1000500
    # obos is first cv element in mzml node
    for o in obos:
        cv_id = o.attrib['id']
        if cv_id in obo_ids:
            continue
        gr = None
        try:
            gr = obonet.read_obo(o.attrib['URI'])
        except Exception:
            if cv_id in DEFAULT_PATH:
                gr = obonet.read_obo(DEFAULT_PATH[cv_id])
            else:
                continue
        if gr is None:
            continue
        gv = gr.nodes(data=True)
        map_ = {
            node_id: {
                "name": data.get("name"),
                "def": data.get("def"),
                "is_a": data.get("is_a")
            }
            for node_id, data in gv
        }
        obo_ids.update(map_)
    return obo_ids


def _children(xr):
    return [re.sub('\{.*\}', '', x.tag) for x in list(xr)]


def _node_attr_recurse(s, ii=None, d=3, c=0,  pre=0):
    # recursively extracts attributes from node s and with depth d, add label children iterator as prefix
    if c == d: return
    if ii is None: ii = []
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


def _vaPar_recurse(s, ii=None, d=9, c=0, dname='accession', dval='value', ddt='all'):
    import re
    # recursively extracts attributes from node s and with depth  d, add label children iterator as prefix
    if c == d: return
    if ii is None: ii = {}
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


#### experiment summary

def _strip_tag(tag):
    return re.sub(r"\{.*\}", "", tag)

def mzml_summary(path, max_spectra=5000):
    def strip(tag):
        return re.sub(r"\{.*\}", "", tag)

    tree = ET.parse(path)
    root = tree.getroot()

    ms_counts = Counter()
    polarities = Counter()
    analysers = set()
    instrument = None
    software = {}
    data_processing = []

    rt_min = None
    rt_max = None

    has_precursor = False
    has_isolation_window = False

    ANALYSER_MAP = {
        "MS:1000084": "TOF",
        "MS:1000484": "Orbitrap",
        "MS:1000081": "Quadrupole",
        "MS:1000264": "Ion Trap",
    }

    # --- instrument info ---
    for elem in root.iter():
        tag = strip(elem.tag)

        if tag == "cvParam":
            name = elem.attrib.get("name", "").lower()

            if "instrument model" in name:
                instrument = elem.attrib.get("name")

        elif tag == "software":
            sid = elem.attrib.get("id")

            entry = {
                "id": sid,
                "version": elem.attrib.get("version"),
                "cv": [],
            }

            for child in elem:
                if _strip_tag(child.tag) == "cvParam":
                    entry["cv"].append({
                        "accession": child.attrib.get("accession"),
                        "name": child.attrib.get("name"),
                        "value": child.attrib.get("value") or None,
                    })

            if sid:
                software[sid] = entry

        elif tag == "processingMethod":
            entry = {
                "order": elem.attrib.get("order"),
                "softwareRef": elem.attrib.get("softwareRef"),
                "cv": [],
            }

            for child in elem:
                if _strip_tag(child.tag) == "cvParam":
                    entry["cv"].append({
                        "accession": child.attrib.get("accession"),
                        "name": child.attrib.get("name"),
                        "value": child.attrib.get("value"),
                    })

            data_processing.append(entry)

        if tag == "spectrum":
            break  # stop once spectra begin

    # --- scan-level info ---
    total_spectra = 0
    no_ms_level = 0

    for spectrum in root.iter():
        if strip(spectrum.tag) != "spectrum":
            continue

        total_spectra += 1
        if total_spectra > max_spectra:
            break

        meta = _vaPar_recurse(spectrum, d=99)

        ms_level = meta.get("MS:1000511")
        if ms_level is None:
            no_ms_level += 1
            continue

        ms_counts[str(ms_level)] += 1

        if "MS:1000130" in meta:
            polarities["positive"] += 1
        elif "MS:1000129" in meta:
            polarities["negative"] += 1

        rt = meta.get("MS:1000016")
        if rt is not None:
            try:
                rt = float(rt)
                if meta.get("UO:0000031") in {"minute", "minutes", "min"}:
                    rt *= 60.0
                rt_min = rt if rt_min is None else min(rt_min, rt)
                rt_max = rt if rt_max is None else max(rt_max, rt)
            except Exception:
                pass

        if "MS:1000744" in meta:
            has_precursor = True

        if {"MS:1000827", "MS:1000828", "MS:1000829"} & meta.keys():
            has_isolation_window = True

    # --- interpretation ---
    has_ms2 = any(int(k) > 1 for k in ms_counts if k.isdigit())

    if not has_ms2:
        acq = "MS1-only"
    elif has_precursor:
        acq = "likely DDA"
    elif has_isolation_window:
        acq = "MS2 with isolation windows"
    else:
        acq = "MS2 present (unknown type)"

    return {
        "file": path,
        "ms_levels": dict(ms_counts),
        "n_spectra_checked": total_spectra,
        "n_no_ms_level": no_ms_level,
        "has_ms2": has_ms2,
        "acquisition": acq,
        "polarity": dict(polarities),
        "instrument": instrument,
        "rt_range_sec": (rt_min, rt_max),
        "analysers": sorted(analysers),
        "software": software,
        "data_processing": data_processing,
    }

def mzxml_summary(path, max_spectra=5000):
    ms_counts = Counter()
    polarities = Counter()
    centroided = Counter()

    rt_min = None
    rt_max = None

    n_scans = 0
    n_with_precursor = 0

    instrument = None
    instrument_info = {}
    analysers = set()
    conversion_software = []


    for event, elem in ET.iterparse(path, events=("start",)):
        tag = _strip_tag(elem.tag)

        # instrument metadata, if present
        if tag in {"msInstrument", "msModel", "msManufacturer", "msIonisation", "msDetector"}:
            instrument_info[tag] = dict(elem.attrib)

            if tag == "msModel":
                instrument = elem.attrib.get("value") or elem.attrib.get("category")

        elif tag == "msMassAnalyzer":
            val = (elem.attrib.get("value") or "").lower()

            if "tof" in val:
                analysers.add("TOF")
            elif "orbitrap" in val:
                analysers.add("Orbitrap")
            elif "quadrupole" in val:
                analysers.add("Quadrupole")
            elif "trap" in val:
                analysers.add("Ion Trap")
        elif tag == "software":
            typ = elem.attrib.get("type", "").lower()
            name = elem.attrib.get("name")
            version = elem.attrib.get("version")

            if typ == "conversion":
                conversion_software.append({
                    "name": name,
                    "version": version
                })


        if tag != "scan":
            continue

        n_scans += 1

        ms_level = elem.attrib.get("msLevel")
        if ms_level is not None:
            ms_counts[ms_level] += 1

        polarity = elem.attrib.get("polarity")
        if polarity == "+":
            polarities["positive"] += 1
        elif polarity == "-":
            polarities["negative"] += 1
        elif polarity is not None:
            polarities[polarity] += 1

        if elem.attrib.get("centroided") == "1":
            centroided["centroided"] += 1
        elif elem.attrib.get("centroided") == "0":
            centroided["profile"] += 1

        rt_iso = elem.attrib.get("retentionTime")
        if rt_iso:
            try:
                rt_sec = isodate.parse_duration(rt_iso).total_seconds()
                rt_min = rt_sec if rt_min is None else min(rt_min, rt_sec)
                rt_max = rt_sec if rt_max is None else max(rt_max, rt_sec)
            except Exception:
                pass

        # mzXML DDA-ish clue: MS2 scans often have precursorMz child,
        # but this is hard to inspect from "start" scan alone.
        if ms_level is not None and int(ms_level) > 1:
            # rough: MS2 exists, detailed precursor handled below if iterating children
            pass

        if n_scans >= max_spectra:
            break

    # second lightweight pass for precursorMz tags
    # because precursorMz is a child element of scan
    for event, elem in ET.iterparse(path, events=("end",)):
        tag = _strip_tag(elem.tag)

        if tag == "precursorMz":
            n_with_precursor += 1

        elem.clear()

    has_ms2 = any(int(k) > 1 for k in ms_counts if str(k).isdigit())

    if not has_ms2:
        acquisition = "MS1-only"
    elif n_with_precursor > 0:
        acquisition = "likely DDA or targeted MS/MS"
    else:
        acquisition = "MS2 present, acquisition type unclear"

    return {
        "file": str(path),
        "format": "mzXML",
        "n_scans_checked": n_scans,
        "ms_levels": dict(ms_counts),
        "has_ms2": has_ms2,
        "acquisition": acquisition,
        "polarity": dict(polarities),
        "spectrum_representation": dict(centroided),
        "rt_range_sec": (rt_min, rt_max),
        "instrument": instrument,
        "instrument_info": instrument_info,
        "analysers": sorted(analysers),
        "n_precursor_mz_tags": n_with_precursor,
        "conversion_software": conversion_software
    }

def inspect_msfile(path, max_spectra=5000):
    suffix = Path(path).suffix.lower()

    if suffix == ".mzml":
        return mzml_summary(path, max_spectra=max_spectra)

    if suffix == ".mzxml":
        return mzxml_summary(path, max_spectra=max_spectra)

    raise ValueError(f"Unsupported file format: {suffix}")

