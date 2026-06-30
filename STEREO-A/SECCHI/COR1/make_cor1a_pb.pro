;==============================================================================
; make_cor1a_pb_local.pro
;
; Local-only SSWIDL script for STEREO-A/SECCHI/COR1-A pB preparation.
;
; Flow:
;   1. Read local COR1-A science FITS files (*_n4c1A.fts or *.fts.gz).
;   2. Find local polarization triplets with POLAR/POLANGL/POLANGLE = 0,120,240 deg.
;   3. Calibrate each image with secchi_prep.
;   4. Build a pB image from the calibrated triplet.
;   5. Write COR1A_pb_pre_<YYYYMMDD>_<HHMMSS>.fits.
;
; Default directories:
;   RAW_DIR = /mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata
;   OUT_DIR = /mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata
;
; Usage in SSWIDL:
;   IDL> .run make_cor1a_pb_local.pro
;   IDL> make_cor1a_pb, '2022-06-13T03:00:00', 1.0
;
; To process all local triplets, omit target_time/window_days:
;   IDL> make_cor1a_pb
;==============================================================================

;------------------------------------------------------------------------------
function kinno_path_join, a, b
  compile_opt idl2
  aa = strtrim(a, 2)
  bb = strtrim(b, 2)
  if strlen(aa) eq 0 then return, bb
  if strmid(aa, strlen(aa)-1, 1) eq '/' then return, aa + bb
  return, aa + '/' + bb
end

;------------------------------------------------------------------------------
function kinno_basename, path
  compile_opt idl2
  s = strtrim(path, 2)
  parts = strsplit(s, '/', /extract)
  if n_elements(parts) gt 0 then return, parts[n_elements(parts)-1]
  return, s
end

;------------------------------------------------------------------------------
function kinno_scalar_string, value, default_value
  compile_opt idl2
  if n_elements(value) eq 0 then return, strtrim(string(default_value), 2)
  s = strtrim(string(value[0]), 2)
  if s eq '' then return, strtrim(string(default_value), 2)
  return, s
end

;------------------------------------------------------------------------------
function kinno_2digit, value
  compile_opt idl2
  return, string(long(value), format='(I02.2)')
end

;------------------------------------------------------------------------------
function kinno_4digit, value
  compile_opt idl2
  return, string(long(value), format='(I04.4)')
end

;------------------------------------------------------------------------------
function kinno_time_string_to_jdsec, value
  compile_opt idl2
  if n_elements(value) eq 0 then return, !values.d_nan
  s = strtrim(string(value[0]), 2)
  if strlen(s) lt 8 then return, !values.d_nan

  catch, err
  if err ne 0 then begin
    catch, /cancel
    return, !values.d_nan
  endif

  if strpos(s, '-') eq 4 or strpos(s, '/') eq 4 then begin
    year = long(strmid(s, 0, 4))
    month = long(strmid(s, 5, 2))
    day = long(strmid(s, 8, 2))
    if strlen(s) ge 13 then hour = long(strmid(s, 11, 2)) else hour = 0L
    if strlen(s) ge 16 then minute = long(strmid(s, 14, 2)) else minute = 0L
    if strlen(s) ge 19 then second = double(strmid(s, 17, 2)) else second = 0.0d0
  endif else begin
    year = long(strmid(s, 0, 4))
    month = long(strmid(s, 4, 2))
    day = long(strmid(s, 6, 2))
    hour = 0L & minute = 0L & second = 0.0d0
    if strlen(s) ge 13 then begin
      hour = long(strmid(s, 9, 2))
      minute = long(strmid(s, 11, 2))
      if strlen(s) ge 15 then second = double(strmid(s, 13, 2))
    endif
  endelse

  jd = julday(month, day, year, hour, minute, second)
  catch, /cancel
  return, double(jd) * 86400.0d0
end

;------------------------------------------------------------------------------
function kinno_jdsec_to_ccsds, jdsec
  compile_opt idl2
  if not finite(jdsec) then return, ''
  jd = double(jdsec) / 86400.0d0
  caldat, jd, month, day, year, hour, minute, second
  sec = long(floor(second + 0.5d0))
  if sec ge 60L then sec = 59L
  return, kinno_4digit(year) + '-' + kinno_2digit(month) + '-' + kinno_2digit(day) + 'T' + $
          kinno_2digit(hour) + ':' + kinno_2digit(minute) + ':' + kinno_2digit(sec)
end

;------------------------------------------------------------------------------
function kinno_key_to_struct_tag, key
  compile_opt idl2
  s = strupcase(strtrim(string(key[0]), 2))
  out = ''
  for i=0L, strlen(s)-1L do begin
    c = strmid(s, i, 1)
    if c eq '-' then c = '_'
    out = out + c
  endfor
  return, out
end

;------------------------------------------------------------------------------
function kinno_struct_tag_index, hdr, key
  compile_opt idl2
  if size(hdr, /type) ne 8 then return, -1L
  tags = tag_names(hdr)
  wanted = kinno_key_to_struct_tag(key)
  idx = where(tags eq wanted, count)
  if count gt 0 then return, long(idx[0])
  return, -1L
end

;------------------------------------------------------------------------------
function kinno_header_is_string_array, hdr
  compile_opt idl2
  return, size(hdr, /type) eq 7
end

;------------------------------------------------------------------------------
function kinno_header_is_struct, hdr
  compile_opt idl2
  return, size(hdr, /type) eq 8
end

;------------------------------------------------------------------------------
function kinno_is_fits_name, path
  compile_opt idl2
  b = strlowcase(kinno_basename(path))
  if strlen(b) ge 4 then if strmid(b, strlen(b)-4, 4) eq '.fts' then return, 1b
  if strlen(b) ge 5 then if strmid(b, strlen(b)-5, 5) eq '.fits' then return, 1b
  if strlen(b) ge 7 then if strmid(b, strlen(b)-7, 7) eq '.fts.gz' then return, 1b
  if strlen(b) ge 8 then if strmid(b, strlen(b)-8, 8) eq '.fits.gz' then return, 1b
  return, 0b
end

;------------------------------------------------------------------------------
function kinno_header_value_string, hdr, keys
  compile_opt idl2

  ; SECCHI_PREP returns a SECCHI_HDR_STRUCT, while HEADFITS returns a FITS
  ; header string array.  Support both, because SXPAR only accepts the latter.
  if kinno_header_is_struct(hdr) then begin
    for ik=0, n_elements(keys)-1 do begin
      itag = kinno_struct_tag_index(hdr, keys[ik])
      if itag ge 0 then begin
        catch, err
        if err eq 0 then begin
          val = hdr.(itag)
          catch, /cancel
          if n_elements(val) gt 0 then return, strtrim(string(val[0]), 2)
        endif else begin
          catch, /cancel
        endelse
      endif
    endfor
    return, ''
  endif

  if kinno_header_is_string_array(hdr) then begin
    for ik=0, n_elements(keys)-1 do begin
      catch, err
      if err eq 0 then begin
        val = sxpar(hdr, keys[ik], count=count)
        catch, /cancel
        if count gt 0 then return, strtrim(string(val[0]), 2)
      endif else begin
        catch, /cancel
      endelse
    endfor
  endif

  return, ''
end

;------------------------------------------------------------------------------
function kinno_header_value_float, hdr, keys
  compile_opt idl2

  ; SECCHI_PREP returns a SECCHI_HDR_STRUCT, while HEADFITS returns a FITS
  ; header string array.  Support both, because SXPAR only accepts the latter.
  if kinno_header_is_struct(hdr) then begin
    for ik=0, n_elements(keys)-1 do begin
      itag = kinno_struct_tag_index(hdr, keys[ik])
      if itag ge 0 then begin
        catch, err
        if err eq 0 then begin
          val = hdr.(itag)
          fval = float(val[0])
          catch, /cancel
          return, fval
        endif else begin
          catch, /cancel
        endelse
      endif
    endfor
    return, !values.f_nan
  endif

  if kinno_header_is_string_array(hdr) then begin
    for ik=0, n_elements(keys)-1 do begin
      catch, err
      if err eq 0 then begin
        val = sxpar(hdr, keys[ik], count=count)
        if count gt 0 then begin
          fval = float(val[0])
          catch, /cancel
          return, fval
        endif
        catch, /cancel
      endif else begin
        catch, /cancel
      endelse
    endfor
  endif

  return, !values.f_nan
end

;------------------------------------------------------------------------------
function kinno_cor1_time_tai, hdr
  compile_opt idl2
  t = kinno_header_value_string(hdr, ['DATE-OBS', 'DATE_OBS', 'DATE_AVG', 'DATE-AVG'])
  if strlen(t) eq 0 then return, !values.d_nan
  return, kinno_time_string_to_jdsec(t)
end

;------------------------------------------------------------------------------
function kinno_cor1_polar_angle, hdr
  compile_opt idl2
  return, kinno_header_value_float(hdr, ['POLAR', 'POLANGL', 'POLANGLE', 'POL_ANG'])
end

;------------------------------------------------------------------------------
function kinno_cor1_detector_ok, hdr
  compile_opt idl2
  det = strlowcase(kinno_header_value_string(hdr, ['DETECTOR', 'DETECT', 'DETECTOR1']))
  inst = strlowcase(kinno_header_value_string(hdr, ['INSTRUME', 'INSTRUMENT']))
  tele = strlowcase(kinno_header_value_string(hdr, ['TELESCOP', 'OBSRVTRY', 'SOURCE']))

  if strpos(det, 'cor1') ge 0 then return, 1b
  if strpos(inst, 'cor1') ge 0 then return, 1b
  if strpos(inst, 'secchi') ge 0 and strpos(tele, 'stereo') ge 0 then return, 1b

  ; Do not reject only because a Level 0.5 header has sparse instrument keywords.
  return, 1b
end

;------------------------------------------------------------------------------
pro kinno_sxadd_string_from_hdr, hdr, key, src_hdr, src_keys
  compile_opt idl2
  value = kinno_header_value_string(src_hdr, src_keys)
  if strlen(strtrim(value, 2)) eq 0 then return

  catch, err
  if err eq 0 then begin
    sxaddpar, hdr, key, value
    catch, /cancel
  endif else begin
    catch, /cancel
  endelse
end

;------------------------------------------------------------------------------
pro kinno_sxadd_float_from_hdr, hdr, key, src_hdr, src_keys
  compile_opt idl2
  value = kinno_header_value_float(src_hdr, src_keys)
  if not finite(value) then return

  catch, err
  if err eq 0 then begin
    sxaddpar, hdr, key, value
    catch, /cancel
  endif else begin
    catch, /cancel
  endelse
end

;------------------------------------------------------------------------------
pro kinno_sxdelpar_safe, hdr, key
  compile_opt idl2

  ; Delete one FITS keyword if present.  This avoids relying on SXDELPAR,
  ; because some SolarSoft installations have slightly different helper sets.
  if size(hdr, /type) ne 7 then return
  if n_elements(hdr) le 0 then return

  wanted = strupcase(strtrim(string(key[0]), 2))
  card_keys = strupcase(strtrim(strmid(hdr, 0, 8), 2))
  idx = where(card_keys ne wanted, count)
  if count gt 0 then hdr = hdr[idx]
end

;------------------------------------------------------------------------------
pro kinno_sanitize_float_image_header, hdr
  compile_opt idl2

  ; The output pB image is written as floating-point data.  Do not keep raw
  ; integer-image scaling keywords from the original COR1 Level 0.5 header.
  ; Keeping BZERO=32768 makes Astropy add 32768 to the pB values on read.
  kinno_sxdelpar_safe, hdr, 'BZERO'
  kinno_sxdelpar_safe, hdr, 'BSCALE'
  kinno_sxdelpar_safe, hdr, 'BLANK'

  ; These statistics describe the input polarizer frame, not the derived pB.
  ; They are reset after pB is computed.
  kinno_sxdelpar_safe, hdr, 'DATAMIN'
  kinno_sxdelpar_safe, hdr, 'DATAMAX'
  kinno_sxdelpar_safe, hdr, 'DATAAVG'
  kinno_sxdelpar_safe, hdr, 'DATASIG'
  kinno_sxdelpar_safe, hdr, 'DATAP01'
  kinno_sxdelpar_safe, hdr, 'DATAP10'
  kinno_sxdelpar_safe, hdr, 'DATAP25'
  kinno_sxdelpar_safe, hdr, 'DATAP50'
  kinno_sxdelpar_safe, hdr, 'DATAP75'
  kinno_sxdelpar_safe, hdr, 'DATAP90'
  kinno_sxdelpar_safe, hdr, 'DATAP95'
  kinno_sxdelpar_safe, hdr, 'DATAP98'
  kinno_sxdelpar_safe, hdr, 'DATAP99'
  kinno_sxdelpar_safe, hdr, 'DATAZER'
  kinno_sxdelpar_safe, hdr, 'DATASAT'
  kinno_sxdelpar_safe, hdr, 'DSATVAL'
end

;------------------------------------------------------------------------------
pro kinno_update_float_data_stats, hdr, data
  compile_opt idl2

  valid = where(finite(data), nvalid)
  if nvalid le 0 then return

  vals = double(data[valid])
  sxaddpar, hdr, 'DATAMIN', min(vals)
  sxaddpar, hdr, 'DATAMAX', max(vals)
  sxaddpar, hdr, 'DATAAVG', mean(vals)
  sxaddpar, hdr, 'DATASIG', stddev(vals)
end

;------------------------------------------------------------------------------
function kinno_make_output_fits_header, raw_file, secchi_hdr
  compile_opt idl2

  ; Start from the original FITS string header so that WRITEFITS/SXADDPAR can
  ; operate normally.  Then overwrite the most important calibrated SECCHI
  ; keywords from SECCHI_HDR_STRUCT returned by SECCHI_PREP.
  catch, err
  if err eq 0 then begin
    hdr = headfits(raw_file)
    catch, /cancel
  endif else begin
    catch, /cancel
    hdr = strarr(1)
    hdr[0] = 'END'
  endelse

  kinno_sxadd_string_from_hdr, hdr, 'DATE-OBS', secchi_hdr, ['DATE_OBS', 'DATE-OBS']
  kinno_sxadd_string_from_hdr, hdr, 'DATE-AVG', secchi_hdr, ['DATE_AVG', 'DATE-AVG']
  kinno_sxadd_string_from_hdr, hdr, 'DATE-END', secchi_hdr, ['DATE_END', 'DATE-END']
  kinno_sxadd_string_from_hdr, hdr, 'DETECTOR', secchi_hdr, ['DETECTOR']
  kinno_sxadd_string_from_hdr, hdr, 'INSTRUME', secchi_hdr, ['INSTRUME']
  kinno_sxadd_string_from_hdr, hdr, 'OBSRVTRY', secchi_hdr, ['OBSRVTRY']
  kinno_sxadd_string_from_hdr, hdr, 'TELESCOP', secchi_hdr, ['TELESCOP']
  kinno_sxadd_string_from_hdr, hdr, 'BUNIT', secchi_hdr, ['BUNIT']
  kinno_sxadd_string_from_hdr, hdr, 'CTYPE1', secchi_hdr, ['CTYPE1']
  kinno_sxadd_string_from_hdr, hdr, 'CTYPE2', secchi_hdr, ['CTYPE2']
  kinno_sxadd_string_from_hdr, hdr, 'CUNIT1', secchi_hdr, ['CUNIT1']
  kinno_sxadd_string_from_hdr, hdr, 'CUNIT2', secchi_hdr, ['CUNIT2']

  kinno_sxadd_float_from_hdr, hdr, 'EXPTIME', secchi_hdr, ['EXPTIME']
  kinno_sxadd_float_from_hdr, hdr, 'POLAR', secchi_hdr, ['POLAR']
  kinno_sxadd_float_from_hdr, hdr, 'CALFAC', secchi_hdr, ['CALFAC']
  kinno_sxadd_float_from_hdr, hdr, 'CRPIX1', secchi_hdr, ['CRPIX1']
  kinno_sxadd_float_from_hdr, hdr, 'CRPIX2', secchi_hdr, ['CRPIX2']
  kinno_sxadd_float_from_hdr, hdr, 'CRVAL1', secchi_hdr, ['CRVAL1']
  kinno_sxadd_float_from_hdr, hdr, 'CRVAL2', secchi_hdr, ['CRVAL2']
  kinno_sxadd_float_from_hdr, hdr, 'CDELT1', secchi_hdr, ['CDELT1']
  kinno_sxadd_float_from_hdr, hdr, 'CDELT2', secchi_hdr, ['CDELT2']
  kinno_sxadd_float_from_hdr, hdr, 'CROTA', secchi_hdr, ['CROTA']
  kinno_sxadd_float_from_hdr, hdr, 'PC1_1', secchi_hdr, ['PC1_1']
  kinno_sxadd_float_from_hdr, hdr, 'PC1_2', secchi_hdr, ['PC1_2']
  kinno_sxadd_float_from_hdr, hdr, 'PC2_1', secchi_hdr, ['PC2_1']
  kinno_sxadd_float_from_hdr, hdr, 'PC2_2', secchi_hdr, ['PC2_2']
  kinno_sxadd_float_from_hdr, hdr, 'RSUN', secchi_hdr, ['RSUN']
  kinno_sxadd_float_from_hdr, hdr, 'XCEN', secchi_hdr, ['XCEN']
  kinno_sxadd_float_from_hdr, hdr, 'YCEN', secchi_hdr, ['YCEN']
  kinno_sxadd_float_from_hdr, hdr, 'CRLN_OBS', secchi_hdr, ['CRLN_OBS']
  kinno_sxadd_float_from_hdr, hdr, 'CRLT_OBS', secchi_hdr, ['CRLT_OBS']
  kinno_sxadd_float_from_hdr, hdr, 'HGLN_OBS', secchi_hdr, ['HGLN_OBS']
  kinno_sxadd_float_from_hdr, hdr, 'HGLT_OBS', secchi_hdr, ['HGLT_OBS']
  kinno_sxadd_float_from_hdr, hdr, 'DSUN_OBS', secchi_hdr, ['DSUN_OBS']

  ; The base header came from an integer raw image.  Remove scaling/statistics
  ; that must not be inherited by a derived floating-point pB FITS.
  kinno_sanitize_float_image_header, hdr

  return, hdr
end

;------------------------------------------------------------------------------
function kinno_norm_pol, pol
  compile_opt idl2
  p = pol mod 360.0
  if p lt 0.0 then p += 360.0
  return, p
end

;------------------------------------------------------------------------------
function kinno_pol_diff, pol, target_pol
  compile_opt idl2
  d = abs(kinno_norm_pol(pol) - kinno_norm_pol(target_pol))
  if d gt 180.0 then d = 360.0 - d
  return, d
end

;------------------------------------------------------------------------------
function kinno_closest_pol_index, tai, pol, candidate_indices, target_pol, reference_tai
  compile_opt idl2
  if n_elements(candidate_indices) lt 1 then return, -1L
  best = -1L
  best_score = 1.0d30
  for i=0, n_elements(candidate_indices)-1 do begin
    idx = candidate_indices[i]
    score = double(abs(tai[idx] - reference_tai)) + 0.1d0 * double(kinno_pol_diff(pol[idx], target_pol))
    if score lt best_score then begin
      best_score = score
      best = idx
    endif
  endfor
  return, best
end

;------------------------------------------------------------------------------
function kinno_list_local_cor1_files, raw_dir
  compile_opt idl2
  pats = [kinno_path_join(raw_dir, '*_n4c1A.fts'), $
          kinno_path_join(raw_dir, '*_n4c1A.fts.gz'), $
          kinno_path_join(raw_dir, '*c1A.fts'), $
          kinno_path_join(raw_dir, '*c1A.fts.gz')]

  files = ''
  nfiles = 0L
  for ip=0, n_elements(pats)-1 do begin
    f = file_search(pats[ip], count=nf)
    if nf gt 0 then begin
      for j=0, nf-1 do begin
        base = kinno_basename(f[j])
        if strpos(base, 'COR1A_pb_pre_') ge 0 then continue
        if strpos(strlowcase(base), 'ptbr') ge 0 then continue
        if strpos(strlowcase(base), 'p000') ge 0 then continue
        if strpos(strlowcase(base), 'p120') ge 0 then continue
        if strpos(strlowcase(base), 'p240') ge 0 then continue
        if kinno_is_fits_name(f[j]) then begin
          if nfiles eq 0L then files = f[j] else files = [files, f[j]]
          nfiles += 1L
        endif
      endfor
    endif
  endfor
  if nfiles eq 0L then return, ['']

  ; Remove duplicates caused by overlapping search patterns.
  order = sort(files)
  files = files[order]
  uniq = files[0]
  nunique = 1L
  for i=1L, n_elements(files)-1L do begin
    if files[i] ne files[i-1L] then begin
      uniq = [uniq, files[i]]
      nunique += 1L
    endif
  endfor
  return, uniq
end

;------------------------------------------------------------------------------
function kinno_output_exists_for_time, out_dir, tai0, tolerance_sec=tolerance_sec
  compile_opt idl2
  if n_elements(tolerance_sec) eq 0 then tol = 2.0d0 else tol = double(tolerance_sec[0])
  pats = [kinno_path_join(out_dir, 'COR1A_pb_pre_*.fits'), $
          kinno_path_join(out_dir, 'COR1A_pb_pre_*.fts')]
  for ip=0, n_elements(pats)-1 do begin
    f = file_search(pats[ip], count=nf)
    if nf le 0 then continue
    for i=0L, nf-1L do begin
      base = kinno_basename(f[i])
      pos = strpos(base, 'COR1A_pb_pre_')
      if pos lt 0 then continue
      ymd = strmid(base, pos + 14, 8)
      hms = strmid(base, pos + 23, 6)
      if strlen(ymd) ne 8 or strlen(hms) ne 6 then continue
      tout = kinno_time_string_to_jdsec(ymd + '_' + hms)
      if finite(tout) then if abs(tout - tai0) le tol then return, 1b
    endfor
  endfor
  return, 0b
end

;------------------------------------------------------------------------------
pro make_cor1a_pb_from_triplet, f0, f120, f240, out_dir, out_file=out_file, overwrite=overwrite
  compile_opt idl2

  print, '[INFO] secchi_prep p000: ', f0
  secchi_prep, f0, h0, i0
  print, '[INFO] secchi_prep p120: ', f120
  secchi_prep, f120, h120, i120
  print, '[INFO] secchi_prep p240: ', f240
  secchi_prep, f240, h240, i240

  i0 = double(i0)
  i120 = double(i120)
  i240 = double(i240)

  if total(size(i0) ne size(i120)) ne 0 or total(size(i0) ne size(i240)) ne 0 then begin
    print, '[SKIP] Triplet image sizes are not identical.'
    return
  endif

  ; Three-polarizer Stokes reconstruction for polarizers separated by 120 deg.
  q = (2.0d0/3.0d0) * (2.0d0*i0 - i120 - i240)
  u = (2.0d0/sqrt(3.0d0)) * (i120 - i240)
  pb = sqrt(q*q + u*u)

  tai0 = kinno_cor1_time_tai(h0)
  if finite(tai0) then tstr = kinno_jdsec_to_ccsds(tai0) else tstr = systime(/utc)

  ymd = strmid(tstr, 0, 4) + strmid(tstr, 5, 2) + strmid(tstr, 8, 2)
  hms = strmid(tstr, 11, 2) + strmid(tstr, 14, 2) + strmid(tstr, 17, 2)
  out_file = kinno_path_join(out_dir, 'COR1A_pb_pre_' + ymd + '_' + hms + '.fits')

  if file_test(out_file) then begin
    if keyword_set(overwrite) then begin
      file_delete, out_file
    endif else begin
      print, '[SKIP] Existing pB output: ', out_file
      return
    endelse
  endif

  hdr = kinno_make_output_fits_header(f0, h0)
  sxaddpar, hdr, 'BUNIT', 'pB'
  sxaddpar, hdr, 'PBDERIV', 'secchi_prep+3pol'
  sxaddpar, hdr, 'POL0FILE', kinno_basename(f0)
  sxaddpar, hdr, 'POL120', kinno_basename(f120)
  sxaddpar, hdr, 'POL240', kinno_basename(f240)
  sxaddhist, 'Created by make_cor1a_pb_local.pro.', hdr
  sxaddhist, 'Input images were calibrated with secchi_prep.', hdr
  sxaddhist, 'pB=sqrt(Q^2+U^2), Q=2/3*(2I0-I120-I240), U=2/sqrt(3)*(I120-I240).', hdr
  kinno_update_float_data_stats, hdr, pb

  writefits, out_file, float(pb), hdr
  print, '[OK] COR1A pB: ', out_file
end

;------------------------------------------------------------------------------
pro make_cor1a_pb_from_local_files, raw_dir, out_dir, target_time=target_time, window_days=window_days, max_triplet_dt_sec=max_triplet_dt_sec, overwrite=overwrite
  compile_opt idl2
  if n_elements(max_triplet_dt_sec) eq 0 then trip_dt = 90.0d0 else trip_dt = double(max_triplet_dt_sec[0])

  use_window = 0b
  target = kinno_scalar_string(target_time, '')
  if strlen(target) gt 0 then begin
    tcen = kinno_time_string_to_jdsec(target)
    if finite(tcen) then begin
      if n_elements(window_days) eq 0 then win_days = 1.0d0 else win_days = double(window_days[0])
      tmin = tcen - win_days*86400.0d0
      tmax = tcen + win_days*86400.0d0
      use_window = 1b
    endif
  endif

  files = kinno_list_local_cor1_files(raw_dir)
  if n_elements(files) lt 3 then begin
    print, '[WARN] Fewer than three local COR1 files found in ', raw_dir
    return
  endif

  n = n_elements(files)
  tai = dblarr(n) + !values.d_nan
  pol = fltarr(n) + !values.f_nan
  keep = bytarr(n)

  print, '[INFO] Inspecting local COR1 files: ', n
  for i=0L, n-1L do begin
    catch, err
    if err ne 0 then begin
      catch, /cancel
      print, '[SKIP] Could not read FITS header: ', files[i]
      continue
    endif
    hdr = headfits(files[i])
    if kinno_cor1_detector_ok(hdr) then begin
      tai[i] = kinno_cor1_time_tai(hdr)
      pol[i] = kinno_cor1_polar_angle(hdr)
      if finite(tai[i]) and finite(pol[i]) then begin
        if use_window then begin
          if tai[i] ge tmin and tai[i] le tmax then keep[i] = 1b
        endif else begin
          keep[i] = 1b
        endelse
      endif
    endif
    catch, /cancel
  endfor

  good = where(keep eq 1b, ngood)
  if ngood lt 3 then begin
    print, '[WARN] Fewer than three usable COR1 polarization images after header inspection.'
    return
  endif

  files = files[good]
  tai = tai[good]
  pol = pol[good]
  order = sort(tai)
  files = files[order]
  tai = tai[order]
  pol = pol[order]

  used = bytarr(n_elements(files))
  nout = 0L
  nskip_existing = 0L

  for i=0L, n_elements(files)-1L do begin
    if used[i] then continue

    dt = abs(tai - tai[i])
    win = where(dt le trip_dt, nw)
    if nw lt 3 then continue

    pwin = fltarr(nw)
    for j=0L, nw-1L do pwin[j] = kinno_norm_pol(pol[win[j]])

    i0_candidates = where((abs(pwin - 0.0) lt 12.0) or (abs(pwin - 360.0) lt 12.0), n0)
    i1_candidates = where(abs(pwin - 120.0) lt 12.0, n1)
    i2_candidates = where(abs(pwin - 240.0) lt 12.0, n2)
    if n0 lt 1 or n1 lt 1 or n2 lt 1 then continue

    idx0 = kinno_closest_pol_index(tai, pol, win[i0_candidates], 0.0, tai[i])
    idx120 = kinno_closest_pol_index(tai, pol, win[i1_candidates], 120.0, tai[i])
    idx240 = kinno_closest_pol_index(tai, pol, win[i2_candidates], 240.0, tai[i])
    if idx0 lt 0 or idx120 lt 0 or idx240 lt 0 then continue
    if used[idx0] or used[idx120] or used[idx240] then continue

    if kinno_output_exists_for_time(out_dir, tai[idx0]) and not keyword_set(overwrite) then begin
      print, '[SKIP] Existing pB output near ', kinno_jdsec_to_ccsds(tai[idx0])
      used[[idx0, idx120, idx240]] = 1b
      nskip_existing += 1L
      continue
    endif

    catch, err2
    if err2 ne 0 then begin
      catch, /cancel
      print, '[SKIP] Failed to make pB near ', kinno_jdsec_to_ccsds(tai[i])
      continue
    endif

    make_cor1a_pb_from_triplet, files[idx0], files[idx120], files[idx240], out_dir, out_file=created, overwrite=overwrite
    used[[idx0, idx120, idx240]] = 1b
    nout += 1L
    catch, /cancel
  endfor

  print, '[INFO] COR1A pB files created       : ', nout
  print, '[INFO] Existing COR1A pB files skipped: ', nskip_existing
end

;------------------------------------------------------------------------------
pro make_cor1a_pb, target_time, window_days, raw_dir=raw_dir, out_dir=out_dir, overwrite=overwrite, max_triplet_dt_sec=max_triplet_dt_sec, _extra=extra
  compile_opt idl2
  target = kinno_scalar_string(target_time, '')
  raw = kinno_scalar_string(raw_dir, '/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/Rawdata')
  out = kinno_scalar_string(out_dir, '/mnt/d/wsl/home/kinno-7010/Research_data/STEREO-A/SECCHI/COR1/pB/Rawdata')
  if n_elements(max_triplet_dt_sec) eq 0 then trip_dt = 90.0d0 else trip_dt = double(max_triplet_dt_sec[0])

  file_mkdir, raw
  file_mkdir, out

  print, '------------------------------------------------------------'
  print, '[INFO] COR1A pB local-only preparation'
  if strlen(target) gt 0 then print, '[INFO] target_time = ', target else print, '[INFO] target_time = all local files'
  if n_elements(window_days) gt 0 then print, '[INFO] window_days = ', double(window_days[0])
  print, '[INFO] raw_dir     = ', raw
  print, '[INFO] out_dir     = ', out
  print, '[INFO] max triplet dt [s] = ', trip_dt
  print, '------------------------------------------------------------'

  if strlen(target) gt 0 then begin
    make_cor1a_pb_from_local_files, raw, out, target_time=target, window_days=window_days, max_triplet_dt_sec=trip_dt, overwrite=overwrite
  endif else begin
    make_cor1a_pb_from_local_files, raw, out, max_triplet_dt_sec=trip_dt, overwrite=overwrite
  endelse
end

;==============================================================================
; End of file
;==============================================================================
