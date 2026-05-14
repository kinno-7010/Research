function read_nsolu_sph,filef90, ph0,th0,r0
 sz_ph=0L
 sz_th=0L
 sz_r=0L
 openr,lun,filef90,/get_lun,/f77_unformatted
 readu,lun, sz_ph, sz_th, sz_r
 print,'sz_ph=',sz_ph, ' sz_th=',sz_th, ' sz_r=',sz_r
 ph0=dblarr(sz_ph)
 th0=dblarr(sz_th)
 r0=dblarr(sz_r)
 nsolu=fltarr(sz_ph, sz_th, sz_r)
 readu,lun,ph0,th0,r0
 readu,lun,nsolu
 free_lun,lun
 pmx,nsolu
 return,nsolu
end

pro show_nmap_sph,rin

 if not keyword_set(rin) then Rin=2.5
 dir='../Sph-grid/'
 den1=read_nsolu_sph(dir+'nsolu_p181t91r51_reg0_nwt_m0.001.dat',ph1,th1,r1) ;mu=0.001, tol=0.001 for Reg0=1
 den2=smooth(den1,2) ;noise efficiently removed by smoothing 
 ntomo=read_nsolu_sph(dir+'nsolu_p181t91r51_reg2_nwt_m1.0.dat',ph0,th0,r0) ;with mu=1.0, tol=0.001 and wt=1/pb(r in cm^-3

 crname='CR2098: mu='+['1.0_Reg2wt','0.001_Reg0wt','reg0_2x2smooth']
 cmp_sphnmap,ntomo,Rin,nemap,ph0,th0,r0,ph1,th1,r1,ph1,th1,r1,cmp1=den1,cmp2=den2,crname=crname
end