;+
; NAME:
;   batch_reduce_lasco_c2_level1
;
; PURPOSE:
;   Convert all LASCO-C2 raw/Level-0.5 FITS files in a directory to Level-1
;   using SSW/LASCO reduce_level_1.pro.
;
;   The calibrated output files are renamed using the calibrated DATE-OBS
;   in the following format:
;
;     <YYYYMMDD>-<HHMMSS>_level1.fts
;
;   If an output file already created by this procedure contains the same
;   RAWFILE keyword, the corresponding raw file is skipped.
;
; INPUT:
;   FITS files in:
;     /mnt/d/wsl/home/kinno-7010/Research_data/SOHO/LASCO-C2_rawdata
;
; OUTPUT:
;   Level-1 calibrated FITS files in:
;     /mnt/d/wsl/home/kinno-7010/Research_data/SOHO/LASCO-C2_rawdata/Level-1_cal
;
; NOTE:
;   This procedure assumes that SSWIDL and LASCO routines are available.
;-

;============================================================
; Make a file-name time tag from the calibrated Level-1 header.
; Expected output format:
;   YYYYMMDD-HHMMSS
;============================================================
function lasco_c2_level1_time_tag, hdr

  date_obs = strtrim(sxpar(hdr, 'DATE-OBS'), 2)
  time_obs = strtrim(sxpar(hdr, 'TIME-OBS'), 2)

  if date_obs eq '' then return, ''

  pos_t = strpos(date_obs, 'T')

  if pos_t ge 0 then begin
    date_part = strmid(date_obs, 0, pos_t)
    time_part = strmid(date_obs, pos_t + 1)
  endif else begin
    date_part = date_obs
    time_part = time_obs
  endelse

  if strlen(date_part) lt 10 then return, ''

  yyyy = strmid(date_part, 0, 4)
  mm   = strmid(date_part, 5, 2)
  dd   = strmid(date_part, 8, 2)

  if strlen(time_part) ge 8 then begin
    hh = strmid(time_part, 0, 2)
    mi = strmid(time_part, 3, 2)
    ss = strmid(time_part, 6, 2)
  endif else begin
    hh = '00'
    mi = '00'
    ss = '00'
  endelse

  return, yyyy + mm + dd + '-' + hh + mi + ss

end


;============================================================
; Build a list of raw-file basenames already processed by this
; procedure.  The RAWFILE keyword is added when this procedure
; writes the final renamed Level-1 file.
;============================================================
pro lasco_c2_level1_get_done_list, outdir, done_rawfiles, done_outfiles, ndone

  done_rawfiles = strarr(1)
  done_outfiles = strarr(1)
  ndone = 0L

  l1files = file_search(outdir + '/*_level1.fts', count=n_l1files)
  if n_l1files eq 0 then return

  tmp_raw = strarr(n_l1files)
  tmp_out = strarr(n_l1files)

  for j = 0L, n_l1files - 1L do begin

    catch, read_error

    if read_error ne 0 then begin
      catch, /cancel
      print, 'WARNING: Could not read existing Level-1 file. Skipping header check:'
      print, l1files[j]
      continue
    endif

    dummy = readfits(l1files[j], tmp_hdr)
    catch, /cancel

    rawfile = strtrim(sxpar(tmp_hdr, 'RAWFILE'), 2)

    if rawfile ne '' then begin
      tmp_raw[ndone] = rawfile
      tmp_out[ndone] = l1files[j]
      ndone = ndone + 1L
    endif

  endfor

  if ndone gt 0 then begin
    done_rawfiles = tmp_raw[0:ndone-1]
    done_outfiles = tmp_out[0:ndone-1]
  endif

end


;============================================================
; Main batch procedure
;============================================================
pro batch_reduce_lasco_c2_level1

  ;============================================================
  ; 1. Directory settings
  ;============================================================
  rawdir = '/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/LASCO-C2_rawdata'
  outdir = '/mnt/d/wsl/home/kinno-7010/Research_data/SOHO/LASCO-C2_rawdata/Level-1_cal'

  ;============================================================
  ; 2. LASCO calibration environment
  ;============================================================
  setenv, 'NRL_LIB=/home/kinno-7010/ssw/soho/lasco'

  occfile = filepath('occulter_center.dat', $
                    root_dir=getenv('NRL_LIB'), $
                    subdirectory=['idl', 'convert'])

  if file_test(occfile) eq 0 then begin
    print, 'ERROR: occulter_center.dat was not found.'
    print, 'Checked path: ', occfile
    print, 'Please check NRL_LIB.'
    return
  endif

  ;============================================================
  ; 3. Create output directory
  ;============================================================
  if file_test(outdir, /directory) eq 0 then begin
    file_mkdir, outdir
  endif

  ;============================================================
  ; 4. Search input FITS files
  ;============================================================
  files = file_search(rawdir + '/*.fts', count=nfiles)

  if nfiles eq 0 then begin
    print, 'No .fts files were found in:'
    print, rawdir
    return
  endif

  files = files[sort(files)]

  ;============================================================
  ; 5. Check already processed files
  ;============================================================
  lasco_c2_level1_get_done_list, outdir, done_rawfiles, done_outfiles, ndone

  print, '============================================================'
  print, 'LASCO-C2 Level-1 batch calibration'
  print, 'Input directory : ', rawdir
  print, 'Output directory: ', outdir
  print, 'Number of input files          : ', nfiles
  print, 'Already processed output files : ', ndone
  print, 'Output name format: <YYYYMMDD>-<HHMMSS>_level1.fts'
  print, '============================================================'

  n_success = 0L
  n_failed  = 0L
  n_skipped = 0L
  n_renamed = 0L

  ;============================================================
  ; 6. Loop over all FITS files
  ;============================================================
  for i = 0L, nfiles - 1L do begin

    infile = files[i]
    rawbase = file_basename(infile)
    outfile = ''
    finalfile = ''

    print, ''
    print, '------------------------------------------------------------'
    print, 'Processing ', strtrim(i + 1, 2), ' / ', strtrim(nfiles, 2)
    print, 'Input: ', infile

    ;----------------------------------------------------------
    ; Skip if this raw file was already processed by this code.
    ;----------------------------------------------------------
    if ndone gt 0 then begin
      idx = where(done_rawfiles eq rawbase, nmatch)
      if nmatch gt 0 then begin
        print, 'SKIP: This raw file has already been calibrated.'
        print, 'Existing Level-1 file: ', done_outfiles[idx[0]]
        n_skipped = n_skipped + 1L
        continue
      endif
    endif

    ;----------------------------------------------------------
    ; If an old default reduce_level_1 output already exists,
    ; rename it instead of recalibrating the raw file.
    ; Example: 22862843.fts -> 25862843.fts
    ;----------------------------------------------------------
    if strlen(rawbase) ge 3 then begin
      old_default = outdir + '/' + '25' + strmid(rawbase, 2)
    endif else begin
      old_default = ''
    endelse

    if (old_default ne '') && file_test(old_default) then begin

      catch, old_error

      if old_error ne 0 then begin
        catch, /cancel
        print, 'WARNING: Old default Level-1 file exists but could not be read.'
        print, 'Old file: ', old_default
        print, 'The raw file will be calibrated again.'
      endif else begin

        old_img = readfits(old_default, old_hdr)
        catch, /cancel

        tag = lasco_c2_level1_time_tag(old_hdr)

        if tag ne '' then begin
          finalfile = outdir + '/' + tag + '_level1.fts'

          if file_test(finalfile) then begin
            print, 'SKIP: Final Level-1 file already exists.'
            print, 'Existing file: ', finalfile
            n_skipped = n_skipped + 1L
            continue
          endif

          fxaddpar, old_hdr, 'RAWFILE', rawbase, 'Original raw FITS filename'
          fxaddpar, old_hdr, 'RAWPATH', infile, 'Original raw FITS path'

          writefits, finalfile, old_img, old_hdr
          file_delete, old_default

          print, 'RENAMED existing Level-1 file:'
          print, '  From: ', old_default
          print, '  To  : ', finalfile

          n_renamed = n_renamed + 1L
          n_success = n_success + 1L
          continue
        endif

      endelse

    endif

    ;----------------------------------------------------------
    ; Main calibration
    ;----------------------------------------------------------
    catch, error_status

    if error_status ne 0 then begin
      catch, /cancel

      print, 'FAILED: ', infile
      print, 'IDL error message:'
      print, !ERROR_STATE.MSG

      n_failed = n_failed + 1L
      continue
    endif

    reduce_level_1, infile, hdr_l1, img_l1, $
                    SAVEDIR=outdir, $
                    OUTFILE=outfile, $
                    /RESET

    catch, /cancel

    ;----------------------------------------------------------
    ; Check temporary/default output from reduce_level_1
    ;----------------------------------------------------------
    if strlen(outfile) eq 0 then begin
      print, 'WARNING: OUTFILE was empty.'
      n_failed = n_failed + 1L
      continue
    endif

    if file_test(outfile) eq 0 then begin
      print, 'WARNING: OUTFILE was returned, but the file was not found.'
      print, 'OUTFILE: ', outfile
      n_failed = n_failed + 1L
      continue
    endif

    ;----------------------------------------------------------
    ; Read calibrated file, rename by calibrated DATE-OBS,
    ; and add RAWFILE/RAWPATH keywords for future skip checks.
    ;----------------------------------------------------------
    catch, rename_error

    if rename_error ne 0 then begin
      catch, /cancel

      print, 'FAILED while renaming calibrated file.'
      print, 'Temporary/default output: ', outfile
      print, 'IDL error message:'
      print, !ERROR_STATE.MSG

      n_failed = n_failed + 1L
      continue
    endif

    img_final = readfits(outfile, hdr_final)
    tag = lasco_c2_level1_time_tag(hdr_final)

    if tag eq '' then begin
      print, 'FAILED: Could not create time tag from calibrated DATE-OBS.'
      print, 'Temporary/default output remains: ', outfile
      n_failed = n_failed + 1L
      catch, /cancel
      continue
    endif

    finalfile = outdir + '/' + tag + '_level1.fts'

    if file_test(finalfile) then begin
      print, 'SKIP: Final Level-1 file already exists.'
      print, 'Existing file: ', finalfile
      print, 'Deleting duplicated temporary/default output: ', outfile
      if outfile ne finalfile then file_delete, outfile
      n_skipped = n_skipped + 1L
      catch, /cancel
      continue
    endif

    fxaddpar, hdr_final, 'RAWFILE', rawbase, 'Original raw FITS filename'
    fxaddpar, hdr_final, 'RAWPATH', infile, 'Original raw FITS path'

    writefits, finalfile, img_final, hdr_final

    if outfile ne finalfile then file_delete, outfile

    catch, /cancel

    print, 'SUCCESS: ', finalfile
    n_success = n_success + 1L

  endfor

  ;============================================================
  ; 7. Summary
  ;============================================================
  print, ''
  print, '============================================================'
  print, 'Batch calibration finished.'
  print, 'Success: ', n_success
  print, 'Renamed existing default outputs: ', n_renamed
  print, 'Skipped: ', n_skipped
  print, 'Failed : ', n_failed
  print, 'Output directory:'
  print, outdir
  print, '============================================================'

end
