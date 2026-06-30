pro get_pbrlc, pmap, rdis, lc, xpa=xpa, width=width, avg=avg,$
  Rmin=Rmin, Rmax=Rmax, err=err, xt=xt,yt=yt,quiet=quiet
;+purpose: extract the light curves along the radial
; direction for average over a certain angular width
; for a certain range of radial distances
; Input: 
;  pmap - polarized brightness for Cor-1 A or B map
; Output:
;  rdis[n]  - radial distance in Rsun 
;  lc[n] - radial pB 
;         if /avg is set, lc is averaged across radius over width
;         and err is standard deviation 
; Keyword
; avg  - if set, average over 2*width+1 pixels perpendicular to radial
; width -in pixel, average over a width for a given angular position
; err - standard deviation for alc if /avg is set
; xpa, - position angle in deg relative to x-axis 
; Rmin  - minimum radial distance in Rsun
; Rmax  -maximum radial distance in Rsun
; History:
; revision on Oct.29,2014: 
;   add treatment that remove the bad pixels in a dark arc-like track
;  near occulter below r=2 Rsun
; revision on Nov.19,2014:
;-
if not keyword_set(width) then width=10 ; width for average across the cut
 ;width=10 pix, about 6 deg at 1.5 Rsun

 ;set value=-100 for pixels inside occulter
   rct=1.5*pmap.rsun/pmap.dx ;assume radius=1.5 Rsun  
   ss=size(pmap.data)
   xdim=ss[1]
   ydim=ss[2]
   xct=(xdim-1)/2.
   yct=(ydim-1)/2.
   x=rebin(reform(findgen(xdim),xdim,1),xdim,ydim)
   y=rebin(reform(findgen(ydim),1,ydim),xdim,ydim)
   d2=(x-xct)^2+(y-yct)^2
   wct=where(d2 le rct^2)
   pmap.data[wct]=-100

 if not keyword_set(quiet) then begin
 !p.multi=0
  loadct,0
  window,1,xs=600, ys=600
  plot_map,pmap,/log,dmin=1.e-10, dmax=2e-8,/limb_plot
  linecolors
 ;print,'== Click Window 1 to continue ...'
 ;cursor,x11,y11,3
 endif


 scl=pmap.dx ;resolution in arcsec
  r2=Rmax*pmap.rsun ;converted to arcsec
  r1=Rmin*pmap.rsun ;minimum radius
  pa=xpa*!pi/180.
  np=round((r2-r1)/scl)+1 ;pixel number between (0,0) and (x0,y0)
  ra=r1+scl*findgen(np)
  xr=ra*cos(pa)
  yr=ra*sin(pa)
  lc=fltarr(np)
  err=fltarr(np)

  if not keyword_set(quiet) then oplot,[0,xr[np-1]],[0,yr[np-1]],color=2

; Average pb(r) over width across a radial cut
  if keyword_set(avg) then begin
     th=pa+!pi/2.
     mm=width*2+1
     xt=fltarr(np,mm)
     yt=fltarr(np,mm)
     for i=0, mm-1 do begin
       xt[*,i]=xr+(i-width)*scl*cos(th)
       yt[*,i]=yr+(i-width)*scl*sin(th)
     endfor
     pxy=fltarr(2,np,mm)
     pxy[0,*,*]=xt
     pxy[1,*,*]=yt
     pos = map_get_pixel(pmap, pxy)
     ix=reform(pos[0,*,*])
     jy=reform(pos[1,*,*])
     lc2d=bilinear(pmap.data,ix,jy)
     err=lc*0.
     for i=0,np-1 do begin
         err[i]=stdev(lc2d[i,*], m0)
         lc[i]=m0
     endfor
  endif $
  else begin ;no average
    pxy=transpose([[xr],[yr]])
   ;pxy[0,*]=xr
   ;pxy[1,*]=yr
    pos = map_get_pixel(pmap, pxy)
    lc=reform(interpolate(pmap.data,pos[0,*],pos[1,*]))
    err=0.1*lc   
  endelse

 ;show pb-light curve along radial cut above occulter
 ; remove bad pixels near occulter
 mlc= max(lc)
 im=where(lc eq mlc)
 for i=0,im[0] do begin
   if lc[i] le mlc-err[im[0]]/2. then lc[i]=-1
 endfor
  
 rdis=ra/pmap.rsun  ;in unit of Rsun
 w=where(lc gt 0)  ;remove negative pixels
 np=n_elements(w)
  rdis=rdis[w]
  lc=lc[w]
  err=err[w]

; remove bad pixels below r=2 Rsun, with values < the mean between r=2-2.5 Rsun
 is=where(rdis gt 2 and rdis lt 2.5)
 m0=average(lc[is])
 nlc=n_elements(lc)
 tag=intarr(nlc)+1
 is=where(rdis le 2, nis)
 for i=0, nis-1 do if lc[i] lt m0 then tag[i]=0
 w=where(tag eq 1)
 np=n_elements(w)
 rdis=rdis[w]
 lc=lc[w]
 err=err[w]


print,'Radial cut distance range:',min(rdis), max(rdis),' Rsun'

 if keyword_set(avg) then begin
       xt=xt[w,*]
       yt=yt[w,*]
       if not keyword_set(quiet) then  $
           for j=0,np-1,10 do oplot,xt[j,*],yt[j,*],color=2
 endif

 if not keyword_set(quiet) then begin
    window,2
    plot,rdis,lc,/nodata,xtitle='Radial distance (Rsun)',$
       ytitle='Polarized intenisty (I/B_mean)',/ylog,$
       xr=[1,4],yr=[1e-10,1.e-7]  
   ;oplot,[1.6,1.6],[1e-10,1e-6]
  endif

 if not keyword_set(quiet) then begin
    if keyword_set(avg) then begin
       iw=indgen(np/10)*10
       e1= lc-err>1e-10
       e2= lc+err>1e-10
      errplot,rdis[iw], e1[iw], e2[iw]
      oplot,rdis,lc,color=2
    endif $
    else oplot,rdis,lc, color=2
   wset,1
   print,'click right button in Window-1 to continue ...'
;  cursor,x0,y0,3
 endif else begin
     if keyword_set(avg) then $
       print,'Averaged relative err for data=',average(err/lc)
 endelse

end
