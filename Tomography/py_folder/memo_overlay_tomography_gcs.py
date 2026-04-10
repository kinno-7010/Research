# main_regularized_tomography.pyで作成したTomogprahy上にGCSを重ね合わせてプロットするスクリプト．
import sys

sys.path.append("/home/kinno-7010/Research_code/GCS")
sys.path.append("/home/kinno-7010/Research_code/GCS/gcs_overlay")
from main_regularized_tomography import *
from gcs_overlay import GCSParams, sample_gcs_wireframe_points
import pyvista as pv

from astropy import units as u
from PyThea.geometrical_models.gcs import GCS

def build_gcs_shell_surface(
    gcs_params,
    n_theta: int = 80,
    n_phi: int = 160,
    tomo_frame: str = "stonyhurst",
    obstime_str: Optional[str] = None,
    observer_for_hgc: str = "earth",
):
    """
    Build a PyVista surface mesh for the GCS shell.

    Returns a PyVista PolyData representing the shell surface.
    """


    # Accept both new (h_apex, lon_deg, ...) and legacy field names for compatibility
    h_apex = getattr(gcs_params, "h_apex", None)
    if h_apex is None:
        h_apex = getattr(gcs_params, "height")

    lon_deg = getattr(gcs_params, "lon_deg", None)
    if lon_deg is None:
        lon_deg = getattr(gcs_params, "longitude")

    lat_deg = getattr(gcs_params, "lat_deg", None)
    if lat_deg is None:
        lat_deg = getattr(gcs_params, "latitude")

    alpha_deg = getattr(gcs_params, "alpha_deg", None)
    if alpha_deg is None:
        alpha_deg = getattr(gcs_params, "half_angle")

    tilt_deg = getattr(gcs_params, "tilt_deg", None)
    if tilt_deg is None:
        tilt_deg = getattr(gcs_params, "tilt")

    shell = GCS(
        height=h_apex * u.R_sun,
        longitude=lon_deg * u.deg,
        latitude=lat_deg * u.deg,
        kappa=getattr(gcs_params, "kappa"),
        half_angle=alpha_deg * u.deg,
        tilt=tilt_deg * u.deg,
        nbverts=max(128, n_phi),  # polyline resolution for internal use
    )

    theta = np.linspace(0.0, np.pi, int(n_theta))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi))
    TT, PP = np.meshgrid(theta, phi, indexing="xy")  # (n_phi, n_theta)

    X, Y, Z = shell.rotate(*shell.shell(TT, PP))  # HGS Cartesian in Rsun

    g = pv.StructuredGrid(X, Y, Z)
    surf = g.extract_surface().triangulate()
    return surf



def main(args, h_apex, kappa, alpha_deg, tilt_deg, lon_deg, lat_deg):
    """
    Run SSC/Ne3dTomo-like preprocessing + regularized tomography WITHOUT argparse.
    Edit the parameters in the `if __name__ == "__main__":` block at the bottom.
    """
    
    params = GCSParams(
        h_apex=h_apex,kappa=kappa,alpha_deg=alpha_deg,
        tilt_deg=tilt_deg,lon_deg=lon_deg,lat_deg=lat_deg
    )
    GCS_STYLE = dict(
        n_parallels=8,
        n_meridians=32,
        color='green',
        color_legs='green',
        lw=1,
        alpha=0.8,
        include_legs=True,
        depth_shade=True,
        alpha_near=0.8,
        alpha_far=0.3,
        alpha_far_legs =0.3,
        leg_depth_from_joint=True,
    )

    defaults = dict(
        pb_fits=[],
        out_n=128,

        default_lonlat="",
        lonlat_file="",

        r_min=1.5,
        r_max=6.0,
        nr=40,
        nth=60,
        nph=120,

        ds=0.02,
        limb_u=DEFAULT_LIMB_U,

        filt=1,
        despike_nsig=6.0,
        despike_med=5,
        pb_floor="",
        dpa_deg=1.0,
        r_use_min=1.5,
        r_use_max=4.0,
        hm=6,
        wt_nr=1,

        lam=1.0,
        q_low=0.0,
        width_pix=2.0,
        maxiter=40,
        tol=1e-4,

        save_prepped_dir="",
        save_ne_npz="",

        show_gui=True,
        freq_mhz=25.0,
        freq_mhz_list=None,
        harmonic=1,
        iso_colors=None,
        save_png=True,
        png_path="",
    )

    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if not args.pb_fits:
        raise ValueError("pb_fits is empty. Set PB_FITS list in the __main__ block.")

    default_lonlat = None
    if args.default_lonlat:
        a, b = args.default_lonlat.split(",")
        default_lonlat = (float(a), float(b))

    lonlat_map = {}
    if args.lonlat_file:
        fp = Path(args.lonlat_file)
        if not fp.exists():
            raise FileNotFoundError(fp)
        import csv
        with fp.open("r", newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#") or len(row) < 3:
                    continue
                lonlat_map[row[0].strip()] = (float(row[1]), float(row[2]))

    pb_paths = [Path(p) for p in args.pb_fits]
    for p in pb_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    pb_overrides = {}
    if args.filt and len(pb_paths) >= 2:
        cube = []
        for p in pb_paths:
            pb0, _ = read_fits_image(p)
            pb1 = block_reduce_mean(pb0, args.out_n) if pb0.shape[0] != args.out_n else pb0
            cube.append(pb1.astype(np.float64))
        cube = despike_pb_cube(np.stack(cube, axis=0), nsig=args.despike_nsig, use_log=True)
        for p, arr in zip(pb_paths, cube):
            pb_overrides[p] = arr

    r_edges = np.linspace(args.r_min, args.r_max, args.nr + 1)
    th_edges = np.linspace(0.0, np.pi, args.nth + 1)
    ph_edges = np.linspace(0.0, 2.0 * np.pi, args.nph + 1)
    grid = SphericalGrid(r_edges=r_edges, th_edges=th_edges, ph_edges=ph_edges)

    save_prepped_dir = Path(args.save_prepped_dir) if args.save_prepped_dir else None

    obs_list: List[Observation] = []
    y_list: List[np.ndarray] = []
    ybk_list: List[Tuple[np.ndarray, np.ndarray]] = []

    for p in pb_paths:
        obs = build_observation(
            pb_fits=p,
            out_n=args.out_n,
            pb_override=pb_overrides.get(p),
            apply_spatial_despike=(p not in pb_overrides),
            r_use_min=args.r_use_min,
            r_use_max=args.r_use_max,
            limb_u=args.limb_u,
            filt=args.filt,
            despike_nsig=args.despike_nsig,
            despike_med=args.despike_med,
            pb_floor=args.pb_floor,
            dpa_deg=args.dpa_deg,
            hm=args.hm,
            width_pix=args.width_pix,
            q_low=args.q_low,
            lonlat_override=lonlat_map.get(p.name) or lonlat_map.get(str(p)) or lonlat_map.get(p.stem),
            lonlat_default=default_lonlat,
            save_prepped_dir=save_prepped_dir,
        )
        obs_list.append(obs)

        y_vec = obs.pb.ravel()[obs.idx_map]
        y_list.append(y_vec)

        rgrid, ybk, _ = ybk_profile_fft(
            pb=obs.pb, hdr=obs.hdr,
            rmin=args.r_use_min, rmax=args.r_use_max,
            dpa_deg=args.dpa_deg, nr=240, hm=args.hm,
            width_pix=args.width_pix, q_low=args.q_low
        )
        ybk_list.append((rgrid, ybk))

        vv = y_vec[np.isfinite(y_vec)]
        if vv.size:
            print(f"[INFO] {p.name}: pB (used pixels) min/med/max = {np.min(vv):.3e} / {np.median(vv):.3e} / {np.max(vv):.3e}")

    y_obs = np.concatenate(y_list) if y_list else np.array([], dtype=float)
    if y_obs.size == 0 or (not np.any(np.isfinite(y_obs))):
        raise ValueError("y_obs is empty or all-NaN. Check masks and preprocessing (r_use_min/max, pb_floor).")

    rays = [build_rays_for_observation(obs=o, grid=grid, ds_rsun=args.ds, r_min=args.r_min, r_max=args.r_max, limb_u=args.limb_u)
            for o in obs_list]

    wt_r = None
    if args.wt_nr:
        r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
        ybks = [np.interp(r_cent, rgi, ybki) for (rgi, ybki) in ybk_list]
        ybk_mean = np.nanmean(np.stack(ybks, axis=0), axis=0)

        good = np.isfinite(ybk_mean) & (ybk_mean > 0)
        if np.count_nonzero(good) < 3:
            print("[WARN] wt_nr requested, but ybk_mean is not usable (too many NaNs). Disabling radial weighting.")
            wt_r = None
        else:
            ybk_clean = ybk_mean.copy()
            if not np.all(good):
                ybk_clean[~good] = np.interp(r_cent[~good], r_cent[good], ybk_mean[good])

            floor = float(np.nanpercentile(ybk_clean[good], 5))
            if not np.isfinite(floor) or floor <= 0:
                floor = float(np.nanmin(ybk_clean[good]))
            floor = max(floor, 1e-30)

            wt_r = 1.0 / np.maximum(ybk_clean, floor)
            wt_r = np.where(np.isfinite(wt_r) & (wt_r > 0), wt_r, 0.0)

    tomo = RegularizedTomography(grid, obs_list, rays, lam=args.lam, wt_r=wt_r)
    ne_raw, info = tomo.solve(y_obs, maxiter=args.maxiter, tol=args.tol, positivity=True)

    if info != 0:
        print(f"[WARN] CG did not fully converge (info={info}). Consider stronger regularization or more images.")

    y_pred = tomo.A_times(ne_raw)
    W = tomo.W
    m = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(W) & (y_pred != 0)
    if np.count_nonzero(m) > 100:
        w2 = (W[m] * W[m])
        num = float(np.sum(w2 * y_pred[m] * y_obs[m]))
        den = float(np.sum(w2 * y_pred[m] * y_pred[m]))
        scale = num / den if den > 0 else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    else:
        scale = 1.0

    ne = ne_raw * scale

    pos = np.isfinite(ne) & (ne > 0)
    if np.any(pos):
        fmin = fp_mhz_from_ne_cm3(float(np.min(ne[pos])), harmonic=args.harmonic)
        fmax = fp_mhz_from_ne_cm3(float(np.max(ne[pos])), harmonic=args.harmonic)
        print(f"[INFO] Reconstructed plasma-frequency range (harm={args.harmonic}): {fmin:.3f} .. {fmax:.3f} MHz")
    else:
        print("[WARN] ne has no positive finite values after scaling.")

    if args.save_ne_npz:
        out = Path(args.save_ne_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            ne=ne.astype(np.float32),
            ne_raw=ne_raw.astype(np.float32),
            scale_brightness=float(scale),
            r_edges=r_edges.astype(np.float32),
            th_edges=th_edges.astype(np.float32),
            ph_edges=ph_edges.astype(np.float32),
        )
        print(f"[OK] Saved solution NPZ: {out}")

    freq_list = list(args.freq_mhz_list) if args.freq_mhz_list is not None else [float(args.freq_mhz)]

    if args.png_path:
        png_path = Path(args.png_path)
    else:
        base = Path(args.save_ne_npz).with_suffix("") if args.save_ne_npz else Path("ne3d_solution")
        tag = "_".join([f"{float(f):.2f}" for f in freq_list])
        png_path = base.parent / f"{base.name}_iso_{tag}MHz_h{int(args.harmonic)}.png"

    print("Save png to", png_path)

    cam_ll = obs_list[0].lonlat_deg if (obs_list and obs_list[0].lonlat_deg) else None

    visualize_isosurface(
        grid=grid,
        ne=ne,
        iso_freqs_mhz=freq_list,
        harmonic=int(args.harmonic),
        show_sun=True,
        opacity=0.5,
        camera_lonlat=cam_ll,
        show_gui=bool(args.show_gui),
        save_png=True,
        png_path=png_path,
        colors=getattr(args, "iso_colors", None),
    )



if __name__ == "__main__":
    from types import SimpleNamespace

    PB_FITS = [
        "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/pB_Kcor_LASCO_axi_20220613_0300.fits",
        "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/COR1A_pb_pre_20220613_030100.fits",
    ]

    DEFAULT_LONLAT = "0.0,0.0"
    LONLAT_FILE = ""

    OUT_N = 128

    # Reconstruction grid
    R_MIN, R_MAX = 2.2, 4.0
    NR, NTH, NPH = 40, 60, 120

    DS = 0.01
    LIMB_U = DEFAULT_LIMB_U

    FILT = 1
    DESPIKE_NSIG = 6.0
    DESPIKE_MED = 5

    PB_FLOOR = ""

    DPA_DEG = 1.0
    R_USE_MIN, R_USE_MAX = 1.5, 4.0
    HM = 6

    WT_NR = 1
    LAM = 1.0
    Q_LOW = 0.0
    WIDTH_PIX = 2.0

    MAXITER = 40
    TOL = 1e-4
    
    # Visualization (isosurfaces specified by plasma frequency)
    SHOW_GUI = True
    HARMONIC = 2

    FREQ_MHZ_LIST = [25.0] #, 31.0, 40.0]
    ISO_COLORS = ["tomato"] #, "deepskyblue", "gold"]
    
    # GCS parameters
    H_APEX = 3.380
    KAPPA = 0.12
    ALPHA_DEG = 20.0
    TILT_DEG = -85.0
    LON_DEG = -44.0
    LAT_DEG = 10.0

    GCS_PARAMS = GCSParams(
        h_apex=H_APEX,
        kappa=KAPPA,
        alpha_deg=ALPHA_DEG,
        tilt_deg=TILT_DEG,
        lon_deg=LON_DEG,
        lat_deg=LAT_DEG,
    )
    
    
    SAVE_PREPPED_DIR = "/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/tomo_prepped"
    SAVE_NE_NPZ = f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/Rawdata/ne3d_solution_"+\
        "-".join(str(f) for f in FREQ_MHZ_LIST)+"MHz.npz"



    SAVE_PNG_PATH = f"/mnt/d/wsl/home/kinno-7010/Research_data/Tomography/output/tomo_" + \
        "-".join(str(f) for f in FREQ_MHZ_LIST) + "MHz.png"

    args = SimpleNamespace(
        pb_fits=PB_FITS,
        out_n=OUT_N,
        default_lonlat=DEFAULT_LONLAT,
        lonlat_file=LONLAT_FILE,

        r_min=R_MIN, r_max=R_MAX, nr=NR, nth=NTH, nph=NPH,
        ds=DS, limb_u=LIMB_U,

        filt=FILT,
        despike_nsig=DESPIKE_NSIG,
        despike_med=DESPIKE_MED,
        pb_floor=PB_FLOOR,
        dpa_deg=DPA_DEG,
        r_use_min=R_USE_MIN,
        r_use_max=R_USE_MAX,
        hm=HM,
        wt_nr=WT_NR,

        lam=LAM,
        q_low=Q_LOW,
        width_pix=WIDTH_PIX,
        maxiter=MAXITER,
        tol=TOL,

        save_prepped_dir=SAVE_PREPPED_DIR,
        save_ne_npz=SAVE_NE_NPZ,

        show_gui=SHOW_GUI,
        freq_mhz_list=FREQ_MHZ_LIST,
        harmonic=HARMONIC,
        iso_colors=ISO_COLORS,

        save_png=True,
        png_path=SAVE_PNG_PATH,
    )

    main(args, H_APEX, KAPPA, ALPHA_DEG, TILT_DEG, LON_DEG, LAT_DEG)
