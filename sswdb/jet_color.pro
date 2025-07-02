;----------------------------------------------------------
; Jetカラーマップ定義（tvlctでカラー適用）
;----------------------------------------------------------
pro jet_color
  r = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0]
  g = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
  b = [0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  tvlct, byte(255*r), byte(255*g), byte(255*b)
end

;----------------------------------------------------------
; プロット処理
;----------------------------------------------------------

; 1. スペクトルデータの転置（周波数×時間）
image = float(transpose(spec))  ; 802×12550（縦×横）

; 2. カラーマップ設定
jet_color

; 3. 時刻軸を hh:mm 文字列に変換
utimes = anytim(tsecs, /vms)
time_labels = tim2str(utimes, /time)

; 4. 時刻ラベルを間引いて取得（例：10個程度にする）
nlabels = 10
idx = round(findgen(nlabels) * (n_elements(tsecs)-1) / (nlabels-1))
x_ticks = idx
x_ticknames = time_labels[x_ticks]

; 5. 軸付きカラープロット
plot_image, image, $
  xtitle='Time (UT)', ytitle='Frequency (MHz)', $
  title='RSTN Learmonth Dynamic Spectrum: ' + date, $
  ycoord=freq, $
  xrange=[0, n_elements(tsecs)-1], $
  /keep_aspect_ratio

; 6. X軸ラベルを手動で設定
axis, xaxis=0, xtickv=x_ticks, xticks=nlabels-1, xtickname=x_ticknames
