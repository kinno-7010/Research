pro sp2xy,Ld,Bd,R,x,y,z,Bd0=Bd0,Pd=Pd
;+Purpose
; transform from Stonyhurst heliographic coord
;(same as Carr.coord. but with Lon. of meridian=0)
; into helicentric-cartesian coord. that is defined
; as: xy in plane of sky with z-axis toward observer,
;   or zx is in equatoral plane with y as solar rotation axis
;Input:  Ld,Bd,R - Lon, Lat (deg), and Rsun (arcsec)
;Output: x,y,z - cartesian coord (arcsec)
; Keyword
; Bd0 - Carrington Latitude of solar disk center 
; Pd  - position angle of solar north, positive when
;   measured counter-clockwise

 if n_elements(Bd0) le 0 then Bd0=0
 if n_elements(Pd) le 0 then Pd=0
 L=Ld*!pi/180.
 B=Bd*!pi/180.
 B0=Bd0*!pi/180.
 P=Pd*!pi/180.
 x1=R*cos(B)*sin(L)
 y1=R*(sin(B)*cos(B0)-cos(B)*cos(L)*sin(B0))
 z1=R*(cos(B)*cos(L)*cos(B0)+sin(B)*sin(B0))
 x=x1*cos(P)-y1*sin(P)
 y=y1*cos(P)+x1*sin(P)
 z=z1
end

pro get_n2d, dens, Ri, nemap, quiet=quiet,iw=iw,name=name,sm=sm, crname=crname
;+purpose: create a spherical cross-sectional map of Ne at solar radius, Rin,
; from the Ne cube in Cartesian coordinate (x,y,z) with z
; along solar rotation axis, and xy for solar equatoral plane,
; Carrington longitude is related to positive x-axis increasing
; counter-clockwise.
; Input:
; dens(128,128,128) - Ne density cube in unit of cm^-3
; Ri - scalar, given radial distance for the synopic map
;     in units of Rsun
; Output:
; nemap(360,181) - synopic map of Ne at Ri
;   I-index for Carrington longtitude (0-359) deg
;   J-index for Carrington latitude (-90 to 90) deg
;-
 if not keyword_set(name) then name=''
 if not keyword_set(iw) then iw=0 
 Rout=4.0d ;outer bound of domain, in unit of Rsun
 Rin =1.5d ;inner bound of damain
 if Ri lt Rin or Ri gt Rout then begin
  print,'The given Ri is out of domain (1.5<r<4)Rsun'
  return
 endif
 sz=size(dens)
 ns=sz[1]
 ds=2.*Rout/double(ns-1) ;size of the cell in the domain

 ;define (lon, lat) coordinates
 ; lon: 0-360 deg, lat: -90 - +90 deg
 lon=findgen(361)
 lon=rebin(reform(lon,361,1), 361,181)
 lat=findgen(181)-90.
 lat=rebin(reform(lat,1,181), 361,181)
 ; convert spheric coord. to Cartesion coord. for projection system
 sp2xy,lon,lat,Ri,xp,yp,zp
 ; convert to Carrington coord. system
 x=zp
 y=xp
 z=yp
  ;convert pixel units
 ix=(x+Rout)/ds ;ix,iy,iz in [0, ns-1]
 iy=(y+Rout)/ds ;
 iz=(z+Rout)/ds ;
 ;interpol points only within domain
   nemap=interpolate(double(dens),ix,iy,iz, missing=0.)
;  nemap=nemap*1.e-6 ;converted to unit of cm^-3 from m^-3
  mne=string(max(nemap),format='(e10.3)')+' (cm!e-3!n)'
  print,mne
 
   m1=1.e5
   m2=1.e7
  ;show result
 if not keyword_set(quiet) then begin
   loadct,0
   print,'sm=',sm
   if keyword_set(sm) then window,iw,xs=600, ys=300,retain=2 $
   else window,iw,xs=800, ys=400,retain=2
   rname=string(Ri, format='(f3.1)')+' Rsun'
   plot_image,alog10(nemap>m1<m2),/noad,xmajor=6,xtickinterval=60.,$
     xtitle='Carrington longitude (deg)',$
     ytitle='Carrington latitude (deg)',$
     title=name+' Ne for '+crname+' at '+ rname,charsize=1.5,$
     origin=[0,-90],scale=[1,1],/xst,/yst
     xyouts,10,80,'Max Ne='+mne,charsize=1.5
 endif

end