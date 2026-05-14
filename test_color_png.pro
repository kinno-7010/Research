pro test_color_png
    compile_opt idl2

    ; ---- 元の描画デバイスを保存 ----
    old_plot = !d.name

    ; ---- PNG保存用に Z-buffer で描画 ----
    set_plot, 'Z'
    device, set_resolution=[512, 512]
    device, set_pixel_depth=24
    device, decomposed=0

    erase
    loadct, 39

    image = findgen(256, 256)
    map   = make_map(image)

    plot_map, map

    ; ---- Z-buffer の内容を取得して PNG 保存 ----
    img = TVRD(/TRUE)
    help, img
    print, min(img), max(img)

    filename = '/home/kinno-7010/Research_code/test_color.png'
    write_png, filename, img

    print, 'FILE_TEST inside proc = ', file_test(filename)

    if file_test(filename) then begin
        img2 = read_png(filename)
        help, img2
        print, 'READ_PNG min/max = ', min(img2), max(img2)
    endif else begin
        print, 'PNG file was not created: ', filename
    endelse

    ; ---- Z-buffer を閉じる ----
    device, /close

    ; ---- GUI 表示用に X に戻して、実際に表示する ----
    if getenv('DISPLAY') ne '' then begin
        set_plot, 'X'

        ; true-color 画像を表示するため一時的に decomposed=1
        device, decomposed=1
        window, /free, xsize=512, ysize=512, title='test_color_png'
        tv, img, true=1

        ; 以後の通常プロット用に戻しておく
        device, decomposed=0
    endif else begin
        set_plot, old_plot
    endelse
end