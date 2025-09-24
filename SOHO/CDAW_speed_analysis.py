'''
SOHO LASCO CDAW CME Speed Analysis Tool
データファイル: 20220613.031211.w360h.v1150.p098g.yht
URL: https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/2022_06/yht/20220613.031211.w360h.v1150.p098g.yht
'''

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.dates as mdates

def read_yht_file(filepath):
    """
    CDAWのYHTファイルを読み込む関数
    """
    metadata = {}
    data_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if ':' in line:
                    key = line.split(':')[0][1:].strip()
                    value = ':'.join(line.split(':')[1:]).strip()
                    metadata[key] = value
            elif line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        height = float(parts[0])
                        date_str = parts[1]
                        time_str = parts[2]
                        angle = float(parts[3])
                        tel = parts[4]
                        fc = int(parts[5])
                        col = float(parts[6]) if len(parts) > 6 else None
                        row = float(parts[7]) if len(parts) > 7 else None
                        
                        datetime_str = f"{date_str} {time_str}"
                        dt = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")
                        
                        data_lines.append({
                            'height': height,
                            'datetime': dt,
                            'angle': angle,
                            'telescope': tel,
                            'feature_code': fc,
                            'col': col,
                            'row': row
                        })
                    except (ValueError, IndexError):
                        continue
    
    df = pd.DataFrame(data_lines)
    return metadata, df

def plot_cme_height_time(metadata, df, output_path=None):
    """
    CME高度-時間プロットを作成する関数
    """
    plt.figure(figsize=(14, 10))
    
    if not df.empty:
        # 時間範囲を03:12:00から04:12:00に限定
        start_time = datetime.strptime("2022/06/13 03:12:00", "%Y/%m/%d %H:%M:%S")
        end_time = datetime.strptime("2022/06/13 04:01:00", "%Y/%m/%d %H:%M:%S")
        
        # データを時間範囲でフィルタリング
        filtered_df = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)].copy()
        
        if filtered_df.empty:
            print("指定された時間範囲にデータがありません")
            return df, metadata
        
        # 時間軸をHH:MM:SS形式で表示するための設定
        time_labels = [dt.strftime('%H:%M:%S') for dt in filtered_df['datetime']]
        
        plt.subplot(2, 2, 1)
        plt.plot(filtered_df['datetime'], filtered_df['height'], 'ro-', markersize=6, linewidth=2)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        plt.xlim(start_time, end_time)
        plt.xlabel('Time (HH:MM:SS)')
        plt.ylabel('Height [$R_{\\odot}$]')
        plt.title(f'CME Height-Time Profile\n{metadata.get("DATE-OBS", "")} {metadata.get("TIME-OBS", "")}')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(filtered_df['height'], filtered_df['angle'], c=filtered_df.index, 
                            s=50, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Time sequence')
        plt.xlabel('Height [$R_{\\odot}$]')
        plt.ylabel('Position Angle [deg]')
        plt.title('Height vs Position Angle (03:12-04:12)')
        plt.grid(True, alpha=0.3)
        
        if len(filtered_df) > 1:
            speeds = []
            speed_times = []
            for i in range(1, len(filtered_df)):
                dt = (filtered_df['datetime'].iloc[i] - filtered_df['datetime'].iloc[i-1]).total_seconds()
                dh = filtered_df['height'].iloc[i] - filtered_df['height'].iloc[i-1]
                if dt > 0:
                    speed = (dh * 6.96e8) / (dt * 1000)  # km/s
                    speeds.append(speed)
                    speed_times.append(filtered_df['datetime'].iloc[i-1] + 
                                     (filtered_df['datetime'].iloc[i] - filtered_df['datetime'].iloc[i-1]) / 2)
            
            plt.subplot(2, 2, 3)
            if speeds:
                plt.plot(speed_times, speeds, 'go-', markersize=5, linewidth=1.5)
                plt.axhline(y=float(metadata.get('SPEED', 0)), color='red', linestyle='--', 
                           label=f'Catalog Speed: {metadata.get("SPEED", "N/A")} km/s')
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
                plt.xlim(start_time, end_time)
                plt.xlabel('Time (HH:MM:SS)')
                plt.ylabel('Instantaneous Speed [km/s]')
                plt.title('CME Speed Evolution (03:12-04:12)')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        tel_colors = {'C2': 'red', 'C3': 'blue'}
        for tel in filtered_df['telescope'].unique():
            tel_data = filtered_df[filtered_df['telescope'] == tel]
            plt.scatter(tel_data['datetime'], tel_data['height'], 
                       c=tel_colors.get(tel, 'gray'), 
                       label=f'LASCO-{tel}', s=50, alpha=0.8)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        plt.xlim(start_time, end_time)
        plt.xlabel('Time (HH:MM:SS)')
        plt.ylabel('Height [R☉]')
        plt.title('Height by Telescope (03:12-04:12)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    info_text = f"""CME Event Information:
Date: {metadata.get('DATE-OBS', 'N/A')} {metadata.get('TIME-OBS', 'N/A')}
Speed: {metadata.get('SPEED', 'N/A')} km/s
Width: {metadata.get('WIDTH', 'N/A')}°
Central PA: {metadata.get('CEN_PA', 'N/A')}
Quality: {metadata.get('QUALITY_INDEX', 'N/A')}
Feature PA: {metadata.get('FEAT_PA', 'N/A')}°
Acceleration: {metadata.get('ACCEL', 'N/A')} m/s²"""
    
    plt.figtext(0.08, 0.3, info_text, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"プロット保存: {output_path}")
    
    plt.show()
    
    return df, metadata

def main():
    """
    メイン実行関数
    """
    data_file = "/mnt/d/wsl/home/kinno-7010/Research/SOHO/LASCO-C2_AIA_image_from_CDAW/20220613.031211.w360h.v1150.p098g.yht"
    
    try:
        print(f"データファイル読み込み中: {data_file}")
        metadata, df = read_yht_file(data_file)
        
        print("\n=== CME Event Metadata ===")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        print(f"\n=== データポイント数: {len(df)} ===")
        if not df.empty:
            print(df.head())
            
            output_file = "/mnt/d/wsl/home/kinno-7010/Research/SOHO/CDAW_cme_analysis_20220613_031211.png"
            plot_cme_height_time(metadata, df, output_file)
        else:
            print("データが見つかりませんでした。")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
