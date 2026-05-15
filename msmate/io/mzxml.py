import numpy as np
import isodate
import xml.etree.cElementTree as ET
import base64
import zlib
import re


def _decode_mzxml_peaks(peaks_node):
    raw = base64.b64decode(peaks_node.text.strip())

    if peaks_node.attrib.get("compressionType", "none") != "none":
        raw = zlib.decompress(raw)

    precision = int(peaks_node.attrib.get("precision", "32"))
    byte_order = peaks_node.attrib.get("byteOrder", "network")

    endian = ">" if byte_order in {"network", "big"} else "<"
    dtype = np.dtype(f"{endian}f{precision // 8}")

    arr = np.frombuffer(raw, dtype=dtype)

    if arr.size % 2:
        raise ValueError("Decoded mzXML peak array has uneven length.")

    arr = arr.reshape(-1, 2)

    return arr[:, 0], arr[:, 1]




def _read_mzxml(self):
    # this is for mzxml version 3.2
    # read in data, files index 0 (see below)
    # flag is 1/2 for msLevel 1 or 2
    tree = ET.parse(self.fpath)
    root = tree.getroot()

    self.out31 = {}

    for scan in root.findall(".//{*}scan"):

        if scan.attrib.get("msLevel") != self.mslevel:
            continue

        st_iso = scan.attrib.get('retentionTime')  # ISO-8601
        st_sec = isodate.parse_duration(st_iso).total_seconds()

        if not self.scan_window.st_max >= st_sec >= self.scan_window.st_min:
            continue

        peaks_node = scan.find("{*}peaks")
        if peaks_node is None or peaks_node.text is None:
            continue

        mz, inten = _decode_mzxml_peaks(peaks_node)

        mask = (mz >= self.scan_window.mz_min) & (mz <= self.scan_window.mz_max)
        mz_filtered = mz[mask]
        inten_filtered = inten[mask]

        scan_num = int(scan.attrib["num"])
        # rt = self._parse_mzxml_rt(scan.attrib["retentionTime"])

        meta = {
            "id": f"scan={scan_num}",
            "index": scan_num,
            "defaultArrayLength": len(mz_filtered),
            "MS:1000016": st_sec,
            "time_unit": "sec",
            "time_unit_ori": "iso8601",
        }

        if scan.attrib.get("polarity") == "-":
            meta["MS:1000129"] = True
        elif scan.attrib.get("polarity") == "+":
            meta["MS:1000130"] = True
        else:
            meta["MS:1000130"] = True  # fallback

        if scan.attrib.get("centroided") == "1":
            meta["MS:1000127"] = True
        else:
            meta["MS:1000128"] = True

        self.out31[scan_num] = {
            "meta": meta,
            "data": {
                "m/z": {"d": mz_filtered},
                "Int": {"d": inten_filtered},
            },
        }

#
# def _parse_mzxml_rt(self, rt: str) -> float:
#     m = re.match(r"PT([0-9.]+)([SMH])", rt)
#
#     if not m:
#         return float(rt)
#
#     value = float(m.group(1))
#     unit = m.group(2)
#
#     if unit == "S":
#         return value
#     if unit == "M":
#         return value * 60
#     if unit == "H":
#         return value * 3600
#
#     return value

