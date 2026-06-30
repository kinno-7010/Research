pro get_cor1_bbk, xr,ybk,sav=sav,sat=sat,name_in=name_in,name_out=name_out,$
   hm=hm
; determine pB background bk(r) from pBmaps for CR2098 using FFT method
; keyword: hm -cutoff frequency. The FFT power for freq>=hm will be set to 0.
 if not keyword_set(hm) then hm=3

restore,name_in
 if keyword_set(pmaps) then times=pmaps.time $
 else if keyword_set(maps) then begin
    pmaps=maps
    times=pmaps.time 
 endif

 if n_elements(ybk) gt 1 then goto, sav

; use pmaps as input to calculate pb[*,*,*]
 cor1_getpbr, time, NR, pB, rh, pa, pmaps, sat=sat, avg=1 

 ss=size(pb)   ;     r,   pa, nimg
 nr=ss[1]
 np=ss[2]
 num=ss[3]
 mbk=fltarr(nr,num)
for k=0,num-1 do begin
 for i=0,nr-1 do begin
    ft=fft(reform(pb[i,*,k]),-1)
    ft[hm:np-hm]=0.
    ;ft[3:np-3]=0. ;when hm=3, keep Fourier freq. m=0, 1, 2, -2, -1
    mbk[i,k]=max(float(fft(ft,1)))
 endfor
endfor
 abk=fltarr(nr)
 aer=fltarr(nr)
 aa=fltarr(nr)
 y1=min(mbk)
 y2=max(mbk)

  apb=pb>0
 for i=0,nr-1 do begin
   aer[i]=stdev(mbk[i,*],m0)
   abk[i]=m0
   aa[i]=average(apb[i,*,*])
 endfor

 window,2,retain=2
 linecolors
;y1=min([aa,y1])
 a2=max(aa)
 b2=max(abk)
 plot,rh,mbk[*,0],/nodata,yr=[1e-10,1e-7],/yst,xtitle='Radial distance (Rsun)',$
  ytitle='pB (Bsun)',/ylog,title='Background For COR1-'+sat

  oplot,rh,abk,color=255,thick=2
  errplot,rh,abk-aer,abk+aer,color=255 
  oplot,rh,aa/a2*b2,color=7,thick=2

  ybk=smooth(abk,5)
;oplot,xr,ybk,color=7
; ybk=smooth(ybk,5)
; xr=1.5+2.5*findgen(200)/199.
; ybk=interpol(ybk,rh,xr)
; w=where(xr gt max(rh),nw)
; ybk[w]=ybk[w[0]-1]
  xr=[rh,4.0]
  ybk=[ybk,ybk[nr-1]]
 oplot,xr,ybk,color=2,thick=2
  
 legend2,['FFT 2-comp BG','Global Avg BG'],psym=-3,color=[2,7]

 sav: print,'save the result-====='
 anw=''
 if keyword_set(sav) then begin
    if keyword_set(name_out) then begin
       result=file_test(name_out)
       if result eq 0 then begin
          save,filename=name_out, pos_obs,pbns, xr,ybk,sat, times 
          print,'file: ['+name_out+'] including [xr,ybk] is saved' 
       endif $
       else begin
          read,name_out+': file alreasy exist! Overwriting?<y/n>!',anw
          if anw eq 'y' then begin
             save,filename=name_out, pos_obs,pbns, xr,ybk,sat, times
             print,'file: ['+name_out+'] including [xr,ybk] is saved'
          endif
       endelse
    endif $
    else begin ;save (xr,ybk) in input file
         save,filename=name_in, pmaps,pos_obs,pbns, xr,ybk,sat, times
          print,'file: ['+name_out+'] including [xr,ybk] is saved'
    endelse
 endif
end
