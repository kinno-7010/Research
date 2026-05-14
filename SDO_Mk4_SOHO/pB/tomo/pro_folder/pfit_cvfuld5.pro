pro pfit_cvfold5,file=file
; determine the best mu by fitting the CV 5-fold ERRs to poly-fit with deg=2
 mfold=5
 derr=fltarr(mfold) 
 if not keyword_set(file) then begin
    print,'Input file is not given! Run as:'
    print,'idl> pfit_cvfold5,file="cv_fold5sp0.20_sz128_nwt.txt"'
    return    
 endif

 openr,1,file
readf,1,n
 mu=fltarr(n)
 av=fltarr(n)
 er=av

 for i=0,n-1 do begin
   readf,1,mu0
   readf,1,derr
;print,'i=',derr
   readf,1,av0
   mu[i]=mu0
   av[i]=av0
   er[i]=sqrt(total((derr-av0)^2)/(mfold-1)) ;=stdev(derr)
 print,'mu=',mu0,' av=',av0,' er=',er[i],stdev(derr)
  endfor
 close,1


linecolors
 lgmu=alog10(mu)

 res=poly_fit(lgmu,av,2,measure_errors=er,sigma=sgm)
 x=min(lgmu)+findgen(100)/99.*(max(lgmu)-min(lgmu))
 y=res[0]+res[1]*x+res[2]*x^2
 w=where(y eq min(y))
 print,'best mu=',10.^x[w],' for min(Err)=',min(y)

 window,0,xs=800,ys=600, retain=2
!p.thick=2
!p.background=255
!p.color=0
!p.charsize=2
 y1=min(av,max=y2)
 x1=min(mu,max=x2)
 plot,mu,av,xtit='Regularization parameter (mu)',ytit='Err (sqrt(model-data)^2)',$
	 /nodata,/xlog, yr=[y1,y2],xr=[x1,x2]
 oplot,mu,av,psym=1
 errplot,mu,av-er,av+er,color=2,width=0.0

 oplot,10.^x,y,color=2
 oplot,10.^[x[w],x[w]],[0,max(y)],color=2
end