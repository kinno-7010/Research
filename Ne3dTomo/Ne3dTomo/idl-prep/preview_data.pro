pro preview_data,sat,cor1_pos,mask,nbin=nbin,filt=filt,gpos=gpos,sav=sav,$
   name_in=name_in,name_out=name_out,quick=quick, nodis=nodis
; preview the data (after rebinned to 128x128) used for 3D density 
;reconstruction using tomography
; if /nbin, then no rebin is applied
; if /filt, despike images
; if /gpos and /sav, get [lon,lat] for sat and save it

 
 if not keyword_set(dmin) then dmin=1.01233e-13
 if not keyword_set(dmax) then dmax=6.95838e-08
   rmn=1e-13

 if not keyword_set(sat) then begin
    print,'sat needs to be given'
    return
 endif

 rin=1.5 ;2.2 ;Rmin for lasco/C2
 rout=4.0 
 tht=findgen(100)/99.*2*!pi
 xp=cos(tht)
 yp=sin(tht)
 restore,name_in ;pmaps,lon,pos_obs,sat,name
 print,'restore data for COR1-'+sat
 pbmaps=pmaps
 ss=size(pbmaps.data)
 nt=ss[3]
 
 if keyword_set(gpos) then begin
         pos_obs=dblarr(3,nt)
      FOR i=0, nt-1 do begin
            pos_obs[*,i]=get_stereo_lonlat(pbmaps[i].time, sat, SYSTEM='CAR', /METERS, /NOVELOCITY)
            pos_obs[1:2,i]=!radeg*pos_obs[1:2,i] ;in degr, Carrington Coord.(Lon, Lat) of observer
            print,'i=',pos_obs[*,i]
      ENDFOR
      print,'Sat-'+sat+' positions are returned!'
 endif
	 

 if not keyword_set(nbin) then isz=128L else isz=512L
 if keyword_set(filt) then begin
   map2index, pbmaps, index, data
   for i=0,nt-1 do begin
	   print,'i=',i
	data[*,*,i]=fix_nan(data[*,*,i]) ;fix NaN pixels
   endfor
   index.date_obs=pbmaps.time
   cleaned=ssw_unspike_cube(index,data,newindex, thresh=6.0) ;6.0
   index2map,newindex,cleaned, pbmaps
   r=data-cleaned
   r=rebin(r,isz,isz,nt)
 endif 

 smap=rebin_map(pbmaps,isz,isz)
 rsun=smap[0].rsun ;in arcsec
 dat=smap.data 
 ;determine the noise level
 y=histogram(dat,binsize=0.2e-10,location=xbin)
 ymx=max(y,im)
 pbns=xbin[im]
 print,'peak freq. pB in histogram=',pbns

   if keyword_set(sav)  then begin
      pmaps=smap
      names=name
      print,'rebin to isz=',isz
      save,filename=name_out,pmaps,pos_obs, pbns, sat,names
      print,'file [',name_out,'] is updated with despiked data and sat-position!'
   endif

IF not keyword_set(nodis) then begin
 set_plot,'x'
 window,1,xs=600,ys=700,retain=2
 !p.multi=[0,1,2,0]
 !p.charsize=1
 plot,xbin,y,/ylog,yr=[0.1,1e5],/xlog,xr=[1e-11,1e-7],$
  xtit='pB',ytit='Pix number'
 plot,xbin,y,/ylog,xr=[0,1e-9],xtit='pB',ytit='Pix number'
  
; show rebinned images
 imgs=bytarr(900,450,nt)
;window,1,xs=500, ys=500
 set_plot,'z'
 device,set_resolution=[900,450]
 !p.multi=0

 ;define image coord. for observer
ix=dindgen(isz)
iy=ix  
; create pix coord. for image
ix = rebin(reform(ix,isz,1), isz,isz)
iy = rebin(reform(iy,1,isz), isz,isz)
c0=dblarr(2,isz,isz)
c0[0,*,*]=ix
c0[1,*,*]=iy
mask=replicate(0b,isz,isz)
 !p.multi=[0,2,1,0]
 for i=0, nt-1 do begin
    map=smap[i]
    p=map_get_coord(map, c0) ;p[2,*,*] in arcsec
    ixp=reform(p[0,*,*])
    iyp=reform(p[1,*,*])
    irp=sqrt(ixp*ixp+iyp*iyp)/map.rsun ;converted in Rsun
    ipw=where(irp gt Rin and irp lt Rout, npw)
    print,'pix number for 128^2 image in 1.5<r<4 Rs=',npw 
    mask[ipw]=1b
    w=where(map.data le pbns)
    ym2=max(map.data)
    ym1=min(map.data)
;   print,'pb min max=',ym1,ym2
;   map.data[w]=ym2 ;ym1
    if i eq 0 then begin
      xr=[min(ixp,max=xr2),xr2]
      yr=[min(iyp,max=yr2),yr2]
    endif
   
    map.data=map.data*mask

  if  keyword_set(quick) then begin  
     if i eq 0 then imgs=bytarr(isz*6,isz*3,nt)
     print,'i=',i,' ',map.time
     imgs[0:3*isz-1,*,i]=bytscl(alog10(rebin(map.data,3*isz,3*isz)>dmin<dmax))
     imgs[3*isz:*,*,i]=bytscl(alog10(rebin(r[*,*,i]>rmn,3*isz,3*isz)))
  endif $
  else begin	  
    loadct,0
    plot_map,map,/log,dmin=dmin,dmax=dmax,xr=xr,yr=yr ;ym1 ;,pbns
;    plot_map,map,/log,dmax=ym2/2.,dmin=0 ;,pbns
  ; linecolors
    x1=xp*rin*rsun  ;[x1,y1] for limb circle at r=1.5 Rsun
    y1=yp*rin*rsun
    x2=xp*rout*rsun ;[x2,y2] for limb circle at r=4.0 Rsun
    y2=yp*rout*rsun
    oplot,x1,y1,thick=2,color=255
    oplot,x2,y2,thick=2,color=255
    if keyword_set(filt) then map.data=r[*,*,i] else map.data=map.data*0+1.
    plot_map,map,/log,xr=xr,yr=yr,tit=' '
    imgs[*,*,i]=tvrd()
  ; cursor,x0,y0,3
  endelse 
 endfor
   set_plot,'x'
   loadct,3
 !p.multi=0
;  linecolors
   gifmov,'tmp.gif',imgs,/mov
ENDIF

end
