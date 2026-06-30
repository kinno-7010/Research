pro pbmap_despike, filt=filt,gpos=gpos,sav=sav,$
   name_in=name_in,name_out=name_out, sat=sat
; preview the data (after rebinned to 128x128) used for 3D density 
;reconstruction using tomography
; if /nbin, then no rebin is applied
; if /filt, despike images
; if /gpos and /sav, get [lon,lat] for sat and save it

;sat='A'


 rin=1.5
 rout=4.0 
 tht=findgen(100)/99.*2*!pi
 xp=cos(tht)
 yp=sin(tht)
 restore,name_in
 print,'restore data for '+sat
 pbmaps=pmaps
 ss=size(pbmaps)
 nt=ss[1]
 
;if keyword_set(sav) then gpos=1
 if keyword_set(gpos) then begin
         pos_obs=dblarr(3,nt) 	 
      FOR i=0, nt-1 do begin
            pos_obs[*,i]=get_stereo_lonlat(pbmaps[i].time, sat, SYSTEM='CAR', /METERS, /NOVELOCITY)
            pos_obs[1:2,i]=!radeg*pos_obs[1:2] ;in degr, Carrington Coord.(Lon, Lat) of observer
            print,'i=',pos_obs[*,i]
      ENDFOR
      print,'Sat-'+sat+' positions are returned!'
 endif

;if not keyword_set(nbin) then isz=128L else isz=512L
  isz=512L
 if keyword_set(filt) then begin
   map2index, pbmaps, index, data
   index.date_obs=pbmaps.time
   cleaned=ssw_unspike_cube(index,data,newindex, thresh=4.)
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
      pbmaps=smap
      print,'rebin to isz=',isz
      save,filename=name_out,pbmaps,pos_obs, pbns
      print,'file [',name_out,'] is updated with despiked data and sat-position!'
   endif


 set_plot,'x'
 window,1,xs=600,ys=700
 !p.multi=[0,1,2,0]
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
    mask[ipw]=1b
    w=where(map.data le pbns)
    ym2=max(map.data)
    ym1=min(map.data)
;   print,'pb min max=',ym1,ym2
;   map.data[w]=ym2 ;ym1
   
    map.data=map.data*mask

    loadct,0
    plot_map,map,/log,dmax=ym2/2.,dmin=0 ;ym1 ;,pbns
  ; linecolors
    x1=xp*rin*rsun  ;[x1,y1] for limb circle at r=1.5 Rsun
    y1=yp*rin*rsun
    x2=xp*rout*rsun ;[x2,y2] for limb circle at r=4.0 Rsun
    y2=yp*rout*rsun
    oplot,x1,y1,thick=2,color=255
    oplot,x2,y2,thick=2,color=255
    if keyword_set(filt) then map.data=r[*,*,i] else map.data=map.data*0+1.
    plot_map,map,/log

    imgs[*,*,i]=tvrd()
  ; cursor,x0,y0,3
 endfor
   set_plot,'x'
   loadct,0
 !p.multi=0
;  linecolors
   gifmov,'tmp.gif',imgs,/mpg 

end
