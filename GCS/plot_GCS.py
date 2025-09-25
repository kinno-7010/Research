"""
run_overlay_example.py
----------------------
Example driver script for overlaying the GCS grid on top of your composite image.

Requirements:
  - Your environment must have SunPy + Astropy installed.
  - Your local `integrated_analysis.py` (the one you uploaded) must be importable,
    along with its `config.py` and data paths properly set on your machine.

How to run (in your environment with data available):
  python run_overlay_example.py "2022-06-13T03:12:00"

This will open a matplotlib window with your existing composite
(AIA+K-Cor+LASCO) and the GCS wireframe grid overlaid.
"""

import sys, os

CANDIDATE_PATHS = [
    # integrated_analysis.py（既に入れているものがあればそのままでOK）
    r"D:\wsl\home\kinno-7010\Research\SDO_Mk4_SOHO\py_folder",
    "/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/py_folder",

    # PyThea リポジトリ（パッケージ PyThea がこの直下にある想定）
    r"D:\wsl\home\kinno-7010\Research\PyThea\Kouloumvakos_GitHub",
    "/mnt/d/wsl/home/kinno-7010/Research/PyThea/Kouloumvakos_GitHub",
]

for p in CANDIDATE_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
from gcs_overlay import GCSParams, overlay_gcs_on_composite

import matplotlib.pyplot as plt
from gcs_overlay.footpoint_fit import find_best_tilt_for_source, find_best_tilt_for_two_sources

# --- Footpoint/tilt fitting controls (optional) ---
FIT_TILT_FROM_SOURCE = False  # set True to enable fitting from source footpoint(s)
SRC1_LONLAT = None            # e.g., (lon_deg, lat_deg)
SRC2_LONLAT = None            # optional 2nd source for opposite leg
TILT_RANGE = (-180.0, 180.0)  # search range [deg]
TILT_STEP  = 1.0              # step [deg]; reduce to 0.2 for fine search



def main(ts: str, h_apex: float, kappa: float, alpha_deg: float, tilt_deg: float, lon_deg: float, lat_deg: float):
    # --- 1) Define your GCS parameters here ---
    # Example values (tune these):
    params = GCSParams(
        h_apex=h_apex,       # [Rsun] apex height
        kappa=kappa,       # minor/major ratio
        alpha_deg=alpha_deg,   # half-angle
        tilt_deg=tilt_deg,    # tilt around axis
        lon_deg=lon_deg,      # HGS longitude
        lat_deg=lat_deg       # HGS latitude
    )

    # --- Optional: auto-fit tilt from source footpoint(s) ---
    if FIT_TILT_FROM_SOURCE and (SRC1_LONLAT is not None):
        try:
            if SRC2_LONLAT is None:
                best = find_best_tilt_for_source(
                    params,
                    SRC1_LONLAT[0],
                    SRC1_LONLAT[1],
                    tilt_search_deg=TILT_RANGE,
                    tilt_step_deg=TILT_STEP,
                    n_phi=360,
                )
                params = params.__class__(**{**params.__dict__, "tilt_deg": best['tilt_deg']})
            else:
                best = find_best_tilt_for_two_sources(
                    params,
                    SRC1_LONLAT,
                    SRC2_LONLAT,
                    tilt_search_deg=TILT_RANGE,
                    tilt_step_deg=TILT_STEP,
                    n_phi=360,
                )
                params = params.__class__(**{**params.__dict__, "tilt_deg": best['tilt_deg']})
        except Exception as e:
            print(f"[WARN] Footpoint-based tilt fit skipped due to error: {e}")

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    res = overlay_gcs_on_composite(
        ax,
        target_time_str=ts,
        gcs_params=params,
        n_parallels=20,  # number of parallels
        n_meridians=20,  # number of meridians
        color='lightgreen',
        lw=1,
        alpha=0.7,
        include_legs=True,
    )
    ax.get_legend()
    ax.set_aspect('equal')
    ax.set_xlabel("X [pixels] ")
    ax.set_ylabel("Y [pixels] ")

    title_time = ts
    base_info = res.get('base') if isinstance(res, dict) else None
    mk4_map = base_info.get('mk4_map') if isinstance(base_info, dict) else None
    if mk4_map is not None:
        try:
            title_time = mk4_map.date.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            title_time = str(mk4_map.date)
    ax.set_title(title_time)
    
    save_path = f"/mnt/d/wsl/home/kinno-7010/Research/GCS/output/GCS_{title_time.replace(':', '')}.png"
    plt.savefig(save_path,dpi=300,bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    h_apex = 2.429               # [Rsun] apex height
    kappa = 0.30               # minor/major ratio
    alpha_deg = 35.0           # half-angle
    tilt_deg = 0.0            # tilt around axis
    lon_deg = -44.0              # HGS longitude
    lat_deg = 21.0              # HGS latitude
    ts = "2022-06-13T03:20:00" # target time

    h_apex_param = float(sys.argv[2]) if len(sys.argv) > 2 else h_apex
    kappa_param = float(sys.argv[3]) if len(sys.argv) > 3 else kappa
    alpha_deg_param = float(sys.argv[4]) if len(sys.argv) > 4 else alpha_deg
    tilt_deg_param = float(sys.argv[5]) if len(sys.argv) > 5 else tilt_deg
    lon_deg_param = float(sys.argv[6]) if len(sys.argv) > 6 else lon_deg
    lat_deg_param = float(sys.argv[7]) if len(sys.argv) > 7 else lat_deg
    ts_param = sys.argv[1] if len(sys.argv) > 1 else ts

# Optional CLI: enable footpoint-based tilt fitting
    # Usage examples:
    #   python plot_GCS.py <ts> <h> <k> <alpha> <tilt> <lon> <lat> fit <lon_src1> <lat_src1> [<lon_src2> <lat_src2>] [<tilt_lo> <tilt_hi> <tilt_step>]
    if len(sys.argv) > 8:
        if str(sys.argv[8]).lower() in ("fit","1","true","yes"):
            FIT_TILT_FROM_SOURCE = True
            if len(sys.argv) > 10:
                SRC1_LONLAT = (float(sys.argv[9]), float(sys.argv[10]))
            if len(sys.argv) > 12:
                SRC2_LONLAT = (float(sys.argv[11]), float(sys.argv[12]))
            if len(sys.argv) > 15:
                TILT_RANGE = (float(sys.argv[13]), float(sys.argv[14]))
                TILT_STEP = float(sys.argv[15])

    main(ts_param, h_apex_param, kappa_param, alpha_deg_param, tilt_deg_param, lon_deg_param, lat_deg_param)
