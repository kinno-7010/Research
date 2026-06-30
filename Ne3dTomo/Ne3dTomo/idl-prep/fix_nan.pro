function fix_nan, in_image
;+ replace NaN pixels in 2D image by average of surrounding pixels
image = FLOAT(in_image)  ; Ensure it's float so NaNs are valid
out=image
 ss=size(image)
 xd=ss[1]
 yd=ss[2]

; Get X, Y positions of all pixels
nx = N_ELEMENTS(image[*,0])
ny = N_ELEMENTS(image[0,*])
x = REBIN(INDGEN(nx), nx, ny)
y = REBIN(TRANSPOSE(INDGEN(ny)), nx, ny)


; Coordinates of NaN pixels
nan = WHERE(~FINITE(image), nan_count)
IF nan_count GT 0 THEN BEGIN
   print,'Number of NaN pixels=',nan_count
  	
  xn = x[nan]
  yn = y[nan]

  ; Interpolate to get new values at NaN locations
   for i=0,nan_count-1 do begin
       if xn[i] gt 0 and xn[i] lt xd-1 and $
	  yn[i] gt 0 and yn[i] lt yd-1 then begin 
	  aa=[image[xn[i]-1,yn[i]], image[xn[i]+1,yn[i]],$
                   image[xn[i],yn[i]-1],image[xn[i],yn[i]+1]]
	  is=where(finite(aa),nf)
	  if nf gt 0 then begin
             out[xn[i],yn[i]]=average(aa[is])
	   print,'NaN is replaced by value=',out[xn[i],yn[i]]
          endif $
	  else begin
		   stop, "all surrounding pixels ar NaN!"
          endelse
       endif
   endfor

  ; Replace NaNs with interpolated values
  print,'NaN pixels in image are fixed!'
  return,out
ENDIF $
ELSE begin
; print,'No NaN pixels are found!' 
  return,out
ENDELSE

end
