pro cmp_sphnmap,ntomo,Ri,nemap,ph0,th0,r0,ph1,th1,r1,ph2,th2,r2,ph3,th3,r3,$
   cmp1=cmp1,cmp2=cmp2,cmp3=cmp3,quiet=quiet,iw=iw,crname=crname,prf=prf,$
   dmin=m1,dmax=m2, c2=c2
;show density map at r=ri from 3D reconstruction using
; spheric grid
if not keyword_set(crname) then begin
	crname=strarr(4)
	crname[*]='CR ????'
endif
 if not keyword_set(iw) then iw=0
 if keyword_set(cmp1) then den1=cmp1
 if keyword_set(cmp2) then den2=cmp2
 if keyword_set(cmp3) then den3=cmp3
 den=ntomo ;cm^-3
 Rout=4.0d ;outer bound of domain, in unit of Rsun
 Rin =1.5d ;inner bound of damain
 if keyword_set(c2) then Rin=2.2 ;for LASCO/C2

 if Ri lt Rin or Ri gt Rout then begin
  print,'The given Ri is out of domain (1.5<r<4.0)Rsun'
  return
 endif
    sz=size(ntomo)
    np=sz[1]
    nt=sz[2]
    nr=sz[3]

 if keyword_set(r0) then begin
     dr=r0[1]-r0[0]
 endif $
 else begin
    dr=(Rout-Rin)/(nr-1)
 endelse
 dp=360./(np-1)
 dt=180./(nt-1)

 i=round((ri-Rin)/dr)
 nemap=den[*,*,i]
 mne=string(max(nemap),format='(e10.3)')+' (cm!e-3!n)'
; print,'N max=',mne

 if not keyword_set(dmin) then  m1=1.e5 
;   m1=1.e5
 if not keyword_set(dmax) then  m2=1.e7 
  ;show result
;if not keyword_set(quiet) then begin
       f=0.8
       loadct,0
       xs=600*f & ys=300*f
      !p.multi=0
      !p.charsize=1.5

   if keyword_set(cmp1) then begin
      ss=size(cmp1)
      np1=ss[1]
      nt1=ss[2]
      dp1=360./(np1-1)
      dt1=180./(nt1-1)
      lon1=ph1*!radeg

      xs=600*f & ys=300*2*f
      !p.multi=[0,1,2,0]
      print,'compare two models'
      !p.charsize=1.5
   endif 

   if keyword_set(cmp2) then begin
      ss=size(cmp2)
      np2=ss[1]
      nt2=ss[2]
      dp2=360./(np2-1)
      dt2=180./(nt2-1)
      lon2=ph2*!radeg

      xs=600*f & ys=300*3*f
      !p.multi=[0,1,3,0]
      print,'compare three models'
      !p.charsize=2
   endif 

   if keyword_set(cmp3) then begin
      ss=size(cmp3)
      np3=ss[1]
      nt3=ss[2]
      dp3=360./(np3-1)
      dt3=180./(nt3-1)
      lon3=ph3*!radeg

      xs=600*f & ys=300*4*f
      !p.multi=[0,1,4,0]
      print,'compare four models'
      !p.charsize=2
   endif


   window,iw,xs=xs, ys=ys,retain=2
   rname=string(Ri, format='(f3.1)')+' Rsun'
   plot_image,alog10(nemap>m1<m2),/noad,xmajor=6,xtickinterval=60.,$
     xtitle='Carrington longitude (deg)',$
     ytitle='Carrington latitude (deg)',$
     title='Ne for '+crname[0]+'  at '+rname,$
     origin=[0,-90],scale=[dp,dt],/xst,/yst
     xyouts,10,80,'Max Ne='+mne,charsize=1.5
     print,'N Model: Nmax=',mne

   if keyword_set(cmp1) then begin
     rname=string(Ri, format='(f3.1)')+' Rsun'
     plot_image,alog10(den1[*,*,i]>m1<m2),/noad,xmajor=6,xtickinterval=60.,$
     xtitle='Carrington longitude (deg)',$
     ytitle='Carrington latitude (deg)',$
     title='Compare 1: Ne for CR '+crname[1]+' at '+rname,$
     origin=[0,-90],scale=[dp1,dt1],/xst,/yst
     mne=string(max(den1[*,*,i]),format='(e10.3)')+' (cm!e-3!n)'
     xyouts,10,80,'Max Ne='+mne,charsize=1.5
     print,'Compare 1: Nmax=',mne
   endif

   if keyword_set(cmp2) then begin
     rname=string(Ri, format='(f3.1)')+' Rsun'
     plot_image,alog10(den2[*,*,i]>m1<m2),/noad,xmajor=6,xtickinterval=60.,$
     xtitle='Carrington longitude (deg)',$
     ytitle='Carrington latitude (deg)',$
     title='Compare 2: Ne for CR '+crname[2]+' at '+rname,$
     origin=[0,-90],scale=[dp2,dt2],/xst,/yst
     mne=string(max(den2[*,*,i]),format='(e10.3)')+' (cm!e-3!n)'
     xyouts,10,80,'Max Ne='+mne,charsize=1.5
     print,'Compare 2: Nmax=',mne
   endif

   if keyword_set(cmp3) then begin
     rname=string(Ri, format='(f3.1)')+' Rsun'
     plot_image,alog10(den3[*,*,i]>m1<m2),/noad,xmajor=6,xtickinterval=60.,$
     xtitle='Carrington longitude (deg)',$
     ytitle='Carrington latitude (deg)',$
     title='Compare 2: Ne for CR '+crname[3]+' at '+rname,$
     origin=[0,-90],scale=[dp3,dt3],/xst,/yst
     mne=string(max(den3[*,*,i]),format='(e10.3)')+' (cm!e-3!n)'
     xyouts,10,80,'Max Ne='+mne,charsize=1.5
     print,'Compare 3: Nmax=',mne
   endif


; endif

  prf=fltarr(nr)
  prf1=prf
  prf2=prf
  prf3=prf
 for k=0,nr-1 do begin
    tmp=den[*,*,k]
    w=where(tmp gt 0, nw)
    prf[k]=total(tmp[w])/nw
    if keyword_set(cmp1) then begin
           tmp=den1[*,*,k]
           w=where(tmp gt 0, nw)
           prf1[k]=total(tmp[w])/nw
    endif
    if keyword_set(cmp2) then begin
           tmp=den2[*,*,k]
           w=where(tmp gt 0, nw)
           prf2[k]=total(tmp[w])/nw
    endif
    if keyword_set(cmp3) then begin
           tmp=den3[*,*,k]
           w=where(tmp gt 0, nw)
           prf3[k]=total(tmp[w])/nw
    endif

   ; prf[i]=average(den[*,*,i])
 endfor
  !p.multi=0

;compare cut along the equator at r=Rin

  linecolors
  window,iw+1,xs=600, ys=400,retain=2
 cut=den[*,nt/2,i]
 lon=ph0*!radeg
 y1=min(cut,max=y2)
 if keyword_set(cmp1) then begin
    cut1=den1[*,nt1/2,i]
    y1=min([cut,cut1],max=y2)
 endif
 if keyword_set(cmp2) then begin
    cut2=den2[*,nt2/2,i]
    y1=min([cut,cut1,cut2],max=y2)
 endif
 if keyword_set(cmp3) then begin
    cut3=den3[*,nt3/2,i]
    y1=min([cut,cut1,cut2,cut3],max=y2)
 endif


 plot,lon,cut,/xst,/yst,thick=1,yr=[y1,y2],$
   charsize=1.4,xtit='Longitude (deg)',ytit='Density (cm^-3)'
   oplot,lon,cut*0,line=2
 if keyword_set(cmp1) then begin
    oplot,lon1,cut1,color=2
 endif
 if keyword_set(cmp2) then begin
    oplot,lon2,cut2,color=7
 endif
 if keyword_set(cmp3) then begin
    oplot,lon3,cut3,color=5
 endif


  window,iw+2,xs=500, ys=400,retain=2
  ;m2=max(prf,min=m1)
  m1=0.5e5
  m2=0.5e7
 rhi=findgen(nr)*dr+Rin
 plot,rhi,prf,yr=[m1,m2],/yst,/ylog,thick=1,$
   charsize=1.4,xtit='Distance (Rsun)',ytit='Density (cm^-3)'
   print,'prf min,max=',min(prf),max(prf)
;linecolors
 if keyword_set(cmp1) then begin
    print,'prf1 min,max=',min(prf1),max(prf1)
    oplot,rhi,prf1,color=2
 ;  legend2,['N','Compare 1'],psym=-3,colors=[255,2],/right
 endif
 if keyword_set(cmp2) then begin
    oplot,rhi,prf2,color=7
    print,'prf2 min,max=',min(prf2),max(prf2)
 ;  legend2,['N','Compare 1','Compare 2'],psym=-3,colors=[255,2,7],/right
 endif
 if keyword_set(cmp3) then begin
    oplot,rhi,prf3,color=5
    print,'prf3 min,max=',min(prf3),max(prf3)
 ; legend2,['N','Compare 1','Compare 2','Compare 3'],psym=-3,colors=[255,2,7,5],/right
 endif
  
 if keyword_set(cmp1) and not keyword_set(cmp2) and not keyword_set(cmp3) then $
     legend2,['N','Compare 1'],psym=-3,colors=[255,2],/right $
 else if keyword_set(cmp2) and not keyword_set(cmp3)  then $
     legend2,['N','Compare 1','Compare 2'],psym=-3,colors=[255,2,7],/right $
 else if keyword_set(cmp3) then $
     legend2,['N','Compare 1','Compare 2','Compare 3'],psym=-3,colors=[255,2,7,5],/right

  !p.multi=0 
end
