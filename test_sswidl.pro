pro test_sswidl
  compile_opt idl2

  print, '=== SSWIDL test start ==='
  print, 'IDL version: ', !version.release
  print, 'OS: ', !version.os_name

  ; 基本計算
  print, '1+1 = ', 1+1

  ; SSW関数が見えるか確認
  x = get_logenv('SSW')
  print, 'SSW env = ', x

  ; パスの先頭だけ表示（長すぎるため一部）
  p = !path
  print, 'PATH(head) = ', strmid(p, 0, 180), '...'

  ; SSW手続きが呼べるか（副作用の少ない確認）
  ssw_addmm_gen, /no_startup
  print, 'ssw_addmm_gen call: OK'

  print, '=== SSWIDL test done ==='
end
