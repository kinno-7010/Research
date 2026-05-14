pro cmp_cartnmap,ntomo,cmp1,cmp2,cmp3,ri=ri,prf=prf,$
	 crname=crname,dmin=m1,dmax=m2, iw=iw
;+show density map at r=ri from 3D reconstruction using
; cartesian grid

 
if not keyword_set(crname) then crname='CR????'
if not keyword_set(iw) then iw=0
if not keyword_set(ri) then Ri=2.5
 if not keyword_set(dmin) then  m1=1.e5
 if not keyword_set(dmax) then  m2=1.e7

 f=0.8
 n0=1.
 loadct,0
 ss=size(ntomo)
 if n_params() eq 1 then begin
	 num=1
	 den=fltarr(ss[1],ss[2],ss[3],num)
	 den[*,*,*,0]=ntomo 
       xs=600*f & ys=300*f
      !p.multi=0
 endif $
 else if n_params() eq 2 then begin
    num=2
    den=fltarr(ss[1],ss[2],ss[3],num)
    den[*,*,*,0]=ntomo
    den[*,*,*,1]=cmp1
      xs=600*f & ys=300*2*f
      !p.multi=[0,1,2,0]
      print,'compare two models'
 endif $
 else if n_params() eq 3 then begin
    num=3
    den=fltarr(ss[1],ss[2],ss[3],num)
    den[*,*,*,0]=n0*ntomo
    den[*,*,*,1]=n0*cmp1
    den[*,*,*,2]=n0*cmp2
       xs=600*f & ys=300*3*f
      !p.multi=[0,1,3,0]
      print,'compare three models'
 endif $
 else if n_params() eq 4 then begin
    num=4
    den=fltarr(ss[1],ss[2],ss[3],num)
    den[*,*,*,0]=n0*ntomo
    den[*,*,*,1]=n0*cmp1
    den[*,*,*,2]=n0*cmp2
    den[*,*,*,3]=n0*cmp3
      xs=600*f & ys=300*4*f
      !p.multi=[0,1,4,0]
      print,'compare three models'
 endif 

 get_n2d, reform(den[*,*,*,0]), ri, map0, iw=iw,crname=crname[0],/quiet 
 ss=size(map0)
 nmap=fltarr(ss[1],ss[2],num)
 nmap[*,*,0]=map0
 for i=1,num-1 do begin
	 get_n2d, reform(den[*,*,*,i]), ri, map0, iw=iw,crname=crname[i],/quiet
	 nmap[*,*,i]=map0
 endfor

 ;show N-map at specified radial height
  window,iw,xs=xs, ys=ys,retain=2
  device,decomposed=0
  dlon=1.
  dlat=1.
  for i=0,num-1 do begin
    rname=string(Ri, format='(f3.1)')+' Rsun'
    img=reform(nmap[*,*,i]>m1<m2)
    mne=string(max(nmap[*,*,i]),format='(e10.3)')+' (cm!e-3!n)'
   plot_image,alog10(img),/noad,xmajor=6,xtickinterval=60.,$
     xtitle='Carrington longitude (deg)',$
     ytitle='Carrington latitude (deg)',$
     title='Ne for CR '+crname[i]+'  at '+rname,charsize=1.5,$
     origin=[0,-90],scale=[dlon,dlat],/xst,/yst, ticklen=-0.02

     xyouts,10,80,'Max Ne='+mne,charsize=1.5
     print,'N Model: Nmax=',mne  
 endfor

 ;show Density profiles along equator at given r=Ri
    linecolors
   !p.multi=0
  window,iw+1,xs=600, ys=400,retain=2
  ss=size(nmap)
  lon=findgen(ss[1])
  cut=reform(nmap[*,ss[2]/2,*])
  y1=min(cut,max=y2)
  clr=[255,2,7,5]
 plot,lon,cut[*,0],/xst,/yst,thick=1,yr=[y1,y2],/nodata,$
   charsize=1.4,xtit='Longitude (deg)',ytit='Density (cm^-3)'
  for i=0,num-1 do begin
    oplot,lon,cut[*,i],color=clr[i]	   
  endfor

;show averaged non-negative density as function of radial distance
if keyword_set(prf) then begin
  dr=0.05
  Rin=1.5
  Rout=4.0
  nr=fix((Rout-Rin)/dr)+1
  rhi=findgen(nr)*dr+Rin
  prf=fltarr(nr,num)
  for j=0,num-1 do begin
     for i=0,nr-1 do begin
       get_n2d, reform(den[*,*,*,j]), rhi[i], map, iw=iw,crname=crname[0],/quiet
          w=where(map gt 0, nw)
          prf[i,j]=total(map[w])/nw
    endfor
  endfor

  window,iw+2,xs=500, ys=400,retain=2
  m1=0.5e5
  m2=0.5e7
  mu=strarr(num)
 plot,rhi,prf[*,0],yr=[m1,m2],/yst,/ylog,thick=1,/nodata,$
   charsize=1.4,xtit='Distance (Rsun)',ytit='Density (cm^-3)'
 for i=0,num-1 do begin
    oplot,rhi,prf[*,i],color=clr[i]
    print,'i=',i,' min,max=',min(prf[*,i]),max(prf[*,i]) 
 endfor
 
legend2,crname,psym=-3,colors=clr[0:num-1],/right
endif

 !p.multi=0
end