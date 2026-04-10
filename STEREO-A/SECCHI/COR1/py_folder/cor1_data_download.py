from pathlib import Path
from datetime import datetime
import requests



def iter_months(start_dt: datetime, end_dt: datetime):
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def parse_summary_text(summary_text: str):
    """
    SECCHI summary の各行から
    (filename, obs_time) を取り出す
    """
    entries = []

    for raw_line in summary_text.splitlines():
        line = raw_line.strip()

        # 空行や区切り線をスキップ
        if not line or line.startswith("=") or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        filename = parts[0]
        date_obs = parts[1]

        # ヘッダ行などを除外
        if filename.lower() == "filename":
            continue

        try:
            obs_time = datetime.strptime(date_obs, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            continue

        entries.append((filename, obs_time))

    return entries


def get_summary_entries(start_dt: datetime, end_dt: datetime, data_type: str):
    """
    data_type: 'seq' or 'img'
    """
    all_entries = []

    for year, month in iter_months(start_dt, end_dt):
        yyyymm = f"{year:04d}{month:02d}"
        summary_name = f"scc{SPACECRAFT_UPPER}{yyyymm}.{data_type}.c1"
        summary_url = f"{BASE_URL}/{SPACECRAFT_LOWER}/summary/{summary_name}"

        print(f"Reading summary: {summary_url}")
        r = requests.get(summary_url, timeout=TIMEOUT)
        r.raise_for_status()

        entries = parse_summary_text(r.text)

        # 時刻で絞り込み
        for filename, obs_time in entries:
            if start_dt <= obs_time <= end_dt:
                all_entries.append((filename, obs_time, data_type))

    return all_entries


def download_one_file(filename: str, obs_time: datetime, data_type: str):
    """
    SECCHI L0 の実ファイルをダウンロード
    例:
    .../L0/a/seq/cor1/20220613/20220613_004500_n4c1A.fts
    """
    yyyymmdd = obs_time.strftime("%Y%m%d")
    file_url = f"{BASE_URL}/{SPACECRAFT_LOWER}/{data_type}/cor1/{yyyymmdd}/{filename}"

    day_dir = SAVE_ROOT / yyyymmdd
    day_dir.mkdir(parents=True, exist_ok=True)

    local_path = day_dir / filename

    if local_path.exists():
        print(f"Skip existing: {local_path}")
        return local_path

    print(f"Downloading: {file_url}")
    with requests.get(file_url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return local_path


def main():
    targets = []

    # 通常の COR1 生データ
    targets.extend(get_summary_entries(START, END, data_type="seq"))

    # 必要なら img 側も拾う
    if INCLUDE_IMG:
        targets.extend(get_summary_entries(START, END, data_type="img"))

    # 重複除去
    seen = set()
    unique_targets = []
    for item in sorted(targets, key=lambda x: x[1]):
        key = (item[0], item[1], item[2])
        if key not in seen:
            seen.add(key)
            unique_targets.append(item)

    print(f"Matched files: {len(unique_targets)}")

    downloaded = []
    for filename, obs_time, data_type in unique_targets:
        local_path = download_one_file(filename, obs_time, data_type)
        downloaded.append(local_path)

    print("\nDownloaded files:")
    for p in downloaded:
        print(p)


if __name__ == "__main__":
    # =========================
    # 設定
    # =========================
    START = datetime(2022, 6, 13, 0, 40, 0)
    END   = datetime(2022, 6, 13, 4, 30, 0)

    SAVE_ROOT = Path("/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    BASE_URL = "https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0"
    SPACECRAFT_UPPER = "A"
    SPACECRAFT_LOWER = "a"

    # COR1 は通常 seq 側
    # 低テレメトリ時の img も拾いたいなら True
    INCLUDE_IMG = False

    TIMEOUT = 60

    main()