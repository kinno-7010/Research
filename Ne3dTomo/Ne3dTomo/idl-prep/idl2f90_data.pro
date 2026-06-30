pro idl2f90_data
 Rsun= 6.957d10 ;cm  solar radius
 isz=128
; convert data for obs_pos, pb_bk, and pb_imgs data into f90 format
  cr=2098
  nc= 1 

 cp=['P1','P2']
 sat='A'
 FOR m=0,nc-1 do begin
   crname='CR'+num2str(cr+m)

  FOR it=0,0 do begin
    crdir='../'+crname+'-'+cp[it]+'/'
    pbname=crdir+'pbmaps_cor1'+sat+'_dspk.sav'
    bkname=crdir+'pbmaps_cor1'+sat+'_pbk.sav'
    restore,pbname
    smap=pmaps
    restore,bkname
    print,'Files['+pbname+','+bkname+'] are restored!'

    f90_cobs=crdir+'cobs_cor1'+sat+'.tex'
    f90_ybk=crdir+'ybk-xr_cor1'+sat+'.tex'
    f90_rbimg=crdir+'pb_cor1'+sat+'_sz128'

    ; for Satellite direction
    openw,1,f90_cobs
    ss=size(pos_obs)
    nt0=ss[2]
    printf,1,nt0
    for k=0,nt0-1 do begin
       printf,1, pos_obs[0,k]*100./Rsun, pos_obs[1,k], pos_obs[2,k],$
        format='(3e23.14)' ;double precision
    endfor
    printf,1,'Lobs (Rsun)    Longitude (deg)   Latitude (deg)'
    close,1
    print,f90_cobs, ' is generated!'

    ;for pB(r) BG
    openw,2,f90_ybk
    nrp=n_elements(ybk)
    printf,2,nrp
    printf,2,xr,ybk
    printf,2,'Radial distance       pB_background'
    close,2
    print,f90_ybk, ' is generated!'

    ;for rebined 128^2 pB images
    for i=0, nt0-1 do begin
       ofile=f90_rbimg+'_'+strcompress(string(i+1, format='(i3.3)'),/remove_all)+'.dat'
       openw,lun,ofile, /GET_LUN, /F77_UNFORMATTED
       writeu,lun, long(isz) ;must be long-type integer for correctly reading in Fortran 77/90
 ; note map.xc/yc/dx/dy/rsun are double-type
       writeu,lun, smap[i].xc, smap[i].yc, smap[i].dx, smap[i].dy, smap[i].rsun
       writeu,lun,smap[i].data ;type of real
       print,'i=',i+1,' min,max=',min(smap[i].data),max(smap[i].data)
       print,smap[i].xc, smap[i].yc, smap[i].dx, smap[i].dy, smap[i].rsun, $
      format='(5d22.14)'
       print,ofile,' is generated-------------'
       free_lun,lun
   endfor

  ENDFOR
  ENDFOR


end
