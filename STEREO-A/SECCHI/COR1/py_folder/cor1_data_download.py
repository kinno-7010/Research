import astropy.units as u
from sunpy.net import Fido, attrs as a

# 検索したい時間範囲を定義
time_range = a.Time('2022-06-13T02:30:00', '2022-06-13T04:30:00')

# 検索クエリを構築
# a.Source('STEREO_A') のように探査機名を文字列として渡すように修正
query = Fido.search(
    time_range,
    a.Instrument.secchi,
    a.Source('STEREO_A'),  # <-- ここを修正しました
    a.Detector.cor1,
    a.Level(1)
)

# 検索結果を表示
print(query)

# 検索結果に合致するファイルをダウンロード
# path引数でダウンロード先のディレクトリを指定
downloaded_files = Fido.fetch(query, path="./stereo_data/")

print(f"Downloaded files: {downloaded_files}")
