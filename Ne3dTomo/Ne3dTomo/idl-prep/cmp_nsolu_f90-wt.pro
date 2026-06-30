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
 return,nsolu
end

pro cmp,Rin,nemap,ph0,th0,r0,ntomo=ntomo
;compare two solutions producted from F90 using readed R_spr 
; and f90-sub created mat_R

 if not keyword_set(rin) then Rin=2.5

 dir='../CR2098-P1/'
 ntomo=read_nsolu_sph(dir+'nsolu_p361t181r51_reg2_nwt.dat',ph0,th0,r0) ;cm^-3; with mu=1.0, tol=0.001 and wt=1/pb(r)

 crname='CR2098: mu='+['COR1A_m1.0','COR1A_m1.0']
cmp_sphnmap, ntomo,Rin,nemap,ph0,th0,r0,ph1,th1,r1,ph2,th2,r2,cmp1=den1,cmp2=den2,crname=crname
end
