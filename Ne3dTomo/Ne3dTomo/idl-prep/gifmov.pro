pro gifmov,name,imgs,info=info,color=color,_extra=_extra,inf1=inf1,$
   eitcolor=eitcolor,zb=zb,aiacolor=aiacolor,nocolor=nocolor,mov=mov,$
   delay=delay,mp4=mp4

ss=size(imgs)
nn=ss(3)
xd=ss(1)
yd=ss(2)

if keyword_set(mov) then goto, showmpg
if keyword_set(nocolor) then goto, nosetclr
if keyword_set(zb) then goto, zbuff

if keyword_set(mp4) then begin
  if keyword_set(info) then begin
    set_plot,'z'
    device,set_resolution=[xd,yd]
    maps=bytarr(xd,yd,nn)
    for k=0,nn-1 do begin
       tv,imgs(*,*,k)
       xyouts,0.01,0.05,info(k),_extra=_extra,/norm,color=255
       if n_elements(inf1) gt 0 then $
          xyouts,0.01,0.95,inf1(k),_extra=_extra,/norm,color=255
       maps[*,*,k]=tvrd()
      ;mask=tvrd() gt 0
     ;imgs(*,*,k)=byte(imgs(*,*,k)*(1-mask)+255*mask) 
   endfor
  endif
  mk_mp4,maps,mvname=name
  return
endif


maps=bytarr(xd,yd,nn)

device,decomposed=0
loadct,0
window,0,xs=xd,ys=yd,xp=200,yp=200
;eis_setup_xwindow,xd,yd,0,zbuff=1

 if n_elements(info) gt 0  and n_elements(color) gt 0 then begin
   for k=0,nn-1 do begin
    erase
    ;time=strmid(info(k),0,18)
    xyouts,0.,0.95,info(k),_extra=_extra,/norm,color=255
    if n_elements(inf1) gt 0 then $
       xyouts,0.,0.85,inf1,_extra=_extra,/norm,color=255
   ;print,time
   map0=tvrd()
   maps(*,*,k)=map0 
  endfor
 endif else $
 if n_elements(info) gt 0  and n_elements(color) le 0 then begin
    for k=0,nn-1 do begin
    ;time=strmid(info(k),0,18)
    tv,imgs(*,*,k)
    xyouts,0.,0.05,info(k),_extra=_extra,/norm,color=255
     if n_elements(inf1) gt 0 then $
       xyouts,0.,0.95,inf1(k),_extra=_extra,/norm,color=255
   ;print,time
   map0=tvrd()
   write_gif,name,map0,/MULTIPLE
  endfor
  write_gif,name,/close
  return
 endif


 window,0,xs=xd,ys=yd,xp=200,yp=200
if n_elements(color) then loadct,color
if n_elements(eitcolor) then eit_colors,195,r,g,b
if n_elements(aiacolor) then aia_lct,wave=aiacolor,/load

;eis_colors,/int
for k=0,nn-1 do begin
  print,'k=',k
  mask=maps(*,*,k) gt 0
  cmap=byte(imgs(*,*,k)*(1-mask)+255*mask)
  tv,cmap
  write_gif,name,cmap,/MULTIPLE 
endfor
  write_gif,name,/close
 goto, showmpg

zbuff: set_plot,'z'
print,'create mov using Z-buff'
device,set_resolution=[xd,yd]
if keyword_set(aiacolor) then aia_lct,wave=171,/load
tvlct,r,g,b,/get
for k=0,nn-1 do begin
  print,'k=',k
; tv,imgs[*,*,k]
; cmap=tvrd()
  cmap=imgs[*,*,k]
  write_gif,name,cmap,r,g,b,/MULTIPLE
endfor
 write_gif,name,/close
 set_plot,'x'

nosetclr: print,'using default color table'
window,0,xs=xd,ys=yd,xp=200,yp=200
for k=0,nn-1 do begin
  print,'k=',k
  cmap=imgs[*,*,k]
  tv,cmap
  tvlct,r,g,b,/get
  if keyword_set(delay) then begin
    write_gif,name,cmap,r,g,b,/MULTIPLE,delay_time=delay
  endif else $
  write_gif,name,cmap,r,g,b,/MULTIPLE
endfor
  write_gif,name,/close

set_plot,'x'
showmpg: print,'show mpg movie--'
    XINTERANIMATE, SET=[xd,yd,nn], /SHOWLOAD
    FOR I=0,nn-1 DO XINTERANIMATE, FRAME = I, IMAGE = imgs[*,*,I]
    XINTERANIMATE

end
