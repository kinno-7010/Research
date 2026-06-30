pro cor1_pbmap, time, pmaps, tmaps, sat=sat, names=names, rotate_on=rotate_on, $
   rsun=rsun, dt_width=dt_width, quiet=quiet,fit=fit,clr=clr,$
   no_calroll=no_calroll,_extra=_extra
;+ derive total brightness and polarized brightness
;from the calibrated images for COR1-A and -B
; Input:
;   time - scalar, given time for pb data, e.g. '2008-02-08 12:00:00'
;   time[2] -1d, starting and end times

; output:
;   pmaps -1 or N pB images in  map-struct
;   tmaps -1 or N total brightness images in map-struct

; keyword:
;   names[3,n]- 1d or 2d, pb-file names for COR1-A or -B
;     if not give, 'time' with keyword 'sat' must be given
;  rotate_on -  rotate Solar North up, default
;  fit - if set using cor1_fitpol, default using cor1_quickpol
;  rsun      - if set, use Rsun as scale unit 
;  dt_width  - time range in minute for searching data when time is scalar. 
;  quiet -  if set not showing plots
;  calroll - default, use combination of monthly min + roll bkg, see scc_getbkgimg.pro
;  no_calroll - if applied, switch off 'calroll', monthly-min-bkg will
;     be applied (default for secchi_prep)
; history:
;  on 2025-0403 add keyword "no_calroll" because some of roll-bkg imgs are not 
;   available in 2011/02
;-

  if not keyword_set(rotate_on) then rotate_on=1
  if not keyword_set(dt_width) then dt_width=2.5 ;min
  
  if not keyword_set(names) then begin
     if not keyword_set(sat) then begin
       print,'Satellite A or B must be specified!'
       return
     endif $
     else begin
       if n_elements(time) eq 1 then begin ; create a time period for search 
     	    utime0=anytim2utc(time)
     	    umin=fix(utime0.time/1000./60.)
    	    utime1=utime0
    	    utime1.time=(umin-dt_width)*60.*1000.>0
            tday=24.*3600.*1000.
            if utime1.time lt 0 then begin
               utime1.mjd=utime1.mjd-1.
               utime1.time=utime1.time+tday
           endif
            time1=utc2str(utime1)
            utime2=utime0
   	    utime2.time=(umin+dt_width)*60.*1000.
            if utime2.time gt tday then begin
               utime2.mjd=utime2.mjd+1.
               utime2.time=utime2.time-tday
            endif
   	    time2=utc2str(utime2)
   	    times=[time1,time2]
	endif else $
        if n_elements(time) eq 2 then times=time $
        else begin
           print,'Time must be a scalar or two-element array!'
           return 
	endelse
        series = cor1_pbseries(times, sat, count=nn)
        if nn eq 0 then begin
          print, 'No data are found in:', times[0],'-',times[1]
          return
       endif else begin
          names=series.filename
          print,'[',nn,'] pB-sequences are found!'
       endelse
     endelse
  endif else begin
    ss=size(names)
    if ss[1] ne 3 then begin
        print,'Names must be [3] or [3, N] array!'
        return
    endif
  endelse
  
  ss=size(names)
  if ss[0] eq 1 then nn=1 $
  else if ss[0] eq 2 then nn=ss[2]

  ; read and calibate the pb-data
 FOR i=0, nn-1 DO BEGIN 
 if not keyword_set(fit) then begin
   print,'---pB is calculated using cor1_quickpol!'
  if not keyword_set(no_calroll) then begin ;use combined monthly minmum bgk + roll bkg 
     SECCHI_PREP, names[*,i],hdr,imgs,/calroll,/interpolate,rotate_on=rotate_on,_extra=_extra
     print,'subtract the combined monthly minmum bgk + roll bkg -----'
  endif $
  else begin   ; use monthly minmum bgk  	  
    SECCHI_PREP, names[*,i],hdr,imgs,/interpolate,rotate_on=rotate_on,_extra=_extra
    print,'subtract the monthly minmum bgk ***************'
  endelse
   htmp=hdr[0]
   cor1_quickpol, imgs, totb, polb
 endif $
 else begin
  print,'---pB is calculated using cor1_fitpol!'
  if not keyword_set(no_calroll) then begin ;use combined monthly minmum bgk + roll bkg 
     SECCHI_PREP, names[*,i],hdr,imgs,/calroll,/interpolate,rotate_on=rotate_on,_extra=_extra
     print,'subtract the combined monthly minmum bgk + roll bkg -----'
  endif $
  else begin   ; use monthly minmum bgk           
    SECCHI_PREP, names[*,i],hdr,imgs,/interpolate,rotate_on=rotate_on,_extra=_extra
    print,'subtract the monthly minmum bgk ***************'
  endelse
  htmp=hdr  ;headers for 0, 120, 240
  cor1_fitpol, htmp, imgs, totb, polb , rotate_on=rotate_on
  ;note, Images must not be rotated before calling this routine
 endelse

 ;converted to map-struct
   wcs=fitshead2wcs(htmp[0])
   wcs2map,totb,wcs,tmap
   wcs2map,polb,wcs,pmap

   tmap.id='COR1-'+sat+' Total brightness'
   pmap.id='COR1-'+sat+' Polarized brightness'

   apos=get_stereo_lonlat(pmap.time, sat, SYSTEM='CAR', /METERS, /NOVELOCITY) ;Carrington 
   lon0=apos[1]*180./!pi  ;Carrington Longitude of 'COR1-A or B'
   tmap.L0=lon0   
   pmap.L0=lon0

  if keyword_set(rsun) then begin  ;convert pixel scale from arcsec to  Rsun
     rs=pmap.rsun
     pmap.xc=pmap.xc/rs
     pmap.yc=pmap.yc/rs
     pmap.dx=pmap.dx/rs
     pmap.dy=pmap.dy/rs
     pmap.xunits='Rsun'
     pmap.yunits='Rsun'

     tmap.xc=tmap.xc/rs
     tmap.yc=tmap.yc/rs
     tmap.dx=tmap.dx/rs
     tmap.dy=tmap.dy/rs
     tmap.xunits='Rsun'
     tmap.yunits='Rsun'
  endif

 if not keyword_set(quiet) then begin
;if not keyword_set(clr) then   loadct,0 else loadct,clr
 eis_colors,/int
    window,1+i
    plot_map,pmap,/log,dmin=1.e-10, dmax=2e-8,/limb_plot
    clon='Carrington Lon for COR1 ='+string(lon0,format='(2X, f7.2)')
    if keyword_set(rsun) then  xyouts, -2,5, clon,charsize=1.5 $
    else xyouts,-4000,5000,clon,charsize=1.5 
 endif
 
 if i eq 0 then begin
    if nn gt 1 then begin
      pmaps=replicate(pmap,nn)
      tmaps=replicate(tmap,nn)
    endif else $
    if nn eq 1 then begin
      pmaps=pmap
      tmaps=tmap
      return
    endif
 endif $
 else begin
;  ;check dimension consistent
   sz=size(pmaps[0].data)
   xms=sz[1]
   sz=size(pmap.data)
   xm1=sz[1]
   if xms ne xm1 then begin
      if xms gt xm1 then begin
         pmaps=rebin_map(pmaps,xm1,xm1)
         tmaps=rebin_map(tmaps,xm1,xm1)
      endif $
      else if xms lt xm1 then begin
         pmap=rebin_map(pmap,xms,xms)
         tmap=rebin_map(tmap,xms,xms)
      endif
      print,'COR1-B: rebinned from ', xms>xm1, ' to ',xms<xm1
   endif
    pmaps[i]=pmap
    tmaps[i]=tmap
 endelse
 ENDFOR

end
