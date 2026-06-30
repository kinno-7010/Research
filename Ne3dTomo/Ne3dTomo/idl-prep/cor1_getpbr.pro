pro cor1_getpbr, time, NR, pB, rh, pa, pmaps, sat=sat,avg=avg,$
   rmin=rmin, rmax=rmax, dpa=dpa,rin=rin, names=names
;+purpose:
; return distribution of pB as a function of positional angle and radial
; distance
; Input pmaps(n)
; output: pB(pa,r)
;-


t0   = systime(1)
 
 if not keyword_set(avg) then avg=1  ;default
 if not keyword_set(Rmin) then Rmin=1.4
 if not keyword_set(Rmax) then Rmax=3.7
 if not keyword_set(dpa)  then dpa=3.0
 if not keyword_set(rin)  then begin
   nrh=100
   rh= 1.5+2.2*findgen(nrh)/(nrh-1)
   rm1=1.5
   rm2=3.7
 endif $
 else begin
   nrh=n_elements(rin)
   rh=rin
 endelse

IF n_elements(pmaps) gt 0 THEN BEGIN
  print,'pBmaps already exist, use it as input-----'
 help,/str,pmaps
ENDIF $
ELSE BEGIN
 if not keyword_set(names) and not keyword_set(sat) then begin
   print,'Satellite A or B must be specified!'
   return
 endif $
 else cor1_pbmap, time, pmaps,names=names, sat=sat, /rotate_on, /quiet
ENDELSE

 nn=n_elements(pmaps)
 print,'[',nn,'] pB maps are read!'
;if keyword_set(nfit) then return

; create angluar positions in [0, 360] deg relative to N-pole, anticlockwise
 npa=fix(360./dpa)
 pa=findgen(npa)*dpa  ;interval=dpa deg
 xpa=pa + 90. ; convert to angels relative to x-axis in [-180,180]
 for i=0, npa-1 do if xpa[i] ge 180. then xpa[i]=xpa[i]-360.

 if nn gt 1 then begin 
   NR=fltarr(nrh, npa, nn)
   pB=fltarr(nrh, npa, nn)
   epb=fltarr(npa,nn) ;relative fitting err for pB
 endif else begin
   NR=fltarr(nrh, npa)
   pB=fltarr(nrh, npa)
   epb=fltarr(npa) ;relative fitting err for pB
 endelse

 M=5  ;polynomial degree for N(r)= sum(A_s/r^s) where s=1-5
; initialize coeff. array for SVD fit
  scl=pmaps[0].dx ;resolution in arcsec
  r2=Rmax*pmaps[0].rsun ;converted to arcsec
  r1=Rmin*pmaps[0].rsun ;minimum radius
  nrr=round((r2-r1)/scl)+1 ;pixel number between (0,0) and (x0,y0)
  rr=Rmin+scl*findgen(nrr)/pmaps[0].rsun
  nk=0

FOR k=0, nn-1 DO BEGIN
  for i=0,npa-1 do begin
    print,'pa=',pa[i], '  deg'
   case avg of 
     1: get_pbrlc, pmaps[k], rdis, lc, err=err_pb, xpa=xpa[i], $
                Rmin=Rmin, Rmax=Rmax ,/avg, /quiet 

     2: get_pbrlc_angle, pmaps[k], rdis, lc, err=err_pb, xpa=xpa[i], dpa=1.5*dpa,$
              Rmin=Rmin, Rmax=Rmax ,/avg , /quiet

     3: get_pbrlc, pmaps[k], rdis, lc, err=err_pb, xpa=xpa[i], $
                Rmin=Rmin, Rmax=Rmax ,avg=0 , /quiet

     else: begin
       print,'keyword [avg] must be set as [1,2,or 3]'
       return
     endelse
   endcase

  pB[*,i,k]=interpol(lc,rdis,rh<max(rdis))    
 endfor

   print,'['+num2str(k+1)+']','in '+num2str(nn)+' is processed'
ENDFOR
print,"CPU TIME = "+string(systime(1)-t0,format='(f10.1)'), ' (seconds)'

end
