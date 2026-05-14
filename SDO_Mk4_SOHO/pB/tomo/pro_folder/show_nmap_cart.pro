function read_nsolu,filef90
 nsz=0L
 openr,lun,filef90,/get_lun,/f77_unformatted
 readu,lun,nsz
 print,'grid size=',nsz
 x0=dblarr(nsz)
 nsolu=fltarr(nsz,nsz,nsz)
 readu,lun,x0
 readu,lun,nsolu
 free_lun,lun
 pmx,nsolu
 return,nsolu
end


pro show_nmap_cart,rin,prf=prf,file=file
;+show the 2D density map at r=rin in 1.5 - 4.0 (Rsun)
 if not keyword_set(rin) then rin=2.5

 dir='../Cart-grid/'
 if keyword_set(file) then $
    den0=read_nsolu(file) else $
  ; den0=read_nsolu(dir+'nsolu_sz128_reg2_nwt_tol0.1.dat')
    den0=read_nsolu('../../V0/Cart-grid/nsolu_sz128_reg2_nwt.dat')
 den1=read_nsolu(dir+'nsolu_sz128_reg2_nwt_tol0.001.dat') ;for comparison

 crname='2098 mu='+['0,3_nwt','0.3_nwt']
 cmp_cartnmap,den0,den1, ri=rin,crname=crname,prf=prf

end
