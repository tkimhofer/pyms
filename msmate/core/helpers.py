from dataclasses import asdict, is_dataclass

def scan_window_to_dict(sw):
    if is_dataclass(sw):
        return asdict(sw)

    return {
        "st_min": sw.st_min,
        "st_max": sw.st_max,
        "mz_min": sw.mz_min,
        "mz_max": sw.mz_max,
    }