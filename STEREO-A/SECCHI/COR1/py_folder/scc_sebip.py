
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Tuple

DIV_MAP = {1:2, 50:4, 53:8, 118:3, 82:2,83:4,84:8,85:16,86:32,87:64,88:128}
SQRT_CODES = {2}

def _parse_ip_codes(hdr: Dict[str, Any]) -> List[int]:
    codes: List[int] = []
    s = hdr.get("IP_00_19") or hdr.get("IP 00 19") or hdr.get("IP-00-19")
    if isinstance(s, (str, bytes)):
        text = s.decode() if isinstance(s, bytes) else s
        for tok in text.strip().split():
            try: codes.append(int(tok))
            except: pass
    for i in range(20):
        key = f"IP_PROG{i}"
        if key in hdr:
            try: codes.append(int(hdr[key]))
            except: pass
    return codes

def scc_sebip(im: np.ndarray, hdr: Dict[str, Any], silent: bool=False) -> Tuple[np.ndarray, Dict[str, Any], int]:
    arr = np.asarray(im, dtype=np.float64)
    codes = _parse_ip_codes(hdr)
    ip_flag = 0
    hist = hdr.get("HISTORY", [])
    if not isinstance(hist, (list, tuple)):
        hist = [hist] if hist else []
    from collections import Counter
    c = Counter(codes)
    for op, div in DIV_MAP.items():
        cnt = c.get(op, 0)
        if cnt > 0:
            factor = float(div) ** cnt
            arr *= factor
            ip_flag = 1
            msg = f"seb_ip Corrected for Divide by {div} x{cnt}"
            hist.append(msg)
            if not silent: print(f"[scc_sebip] {msg}")
    cnt_sqrt = sum(c.get(op, 0) for op in SQRT_CODES)
    if cnt_sqrt > 0:
        arr = np.maximum(arr, 0.0)
        for _ in range(cnt_sqrt):
            arr = arr * arr
        ip_flag = 1
        msg = f"seb_ip Corrected for Square-Root x{cnt_sqrt}"
        hist.append(msg)
        if not silent: print(f"[scc_sebip] {msg}")
    hdr["HISTORY"] = hist
    return arr, hdr, ip_flag
