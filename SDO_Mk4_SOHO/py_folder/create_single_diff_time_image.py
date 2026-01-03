from main_coronagraph_analysis import single_time_analysis_from_min

from config import *
import config
from claude_analysis_utils import (
    analyze_single_time_cme_with_diff_image,
    analyze_single_time_cme_with_diff_from_min_image,
    analyze_single_time_cme_with_raw_image,
    compare_cme_heights_multiple_times,
    run_cme_analysis_workflow
)
from integrated_analysis import create_single_diff_from_time_image, clear_scan_cache, get_cache_info
from cme_measurement import analyze_single_time_cme_multi_points
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.time import Time

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(12, 12))
    target_time_str = "2022-06-13T02:01:00"
    base_time_str = "2022-06-13T01:00:00"
    create_single_diff_from_time_image(ax, target_time_str, base_time_str)
    plt.savefig(f"/mnt/d/wsl/home/kinno-7010/Research/SDO_Mk4_SOHO/CME_measurement/diff_from_time_image/diff_from_time_image_{target_time_str.replace(':', '')}.png")
    plt.show()
    