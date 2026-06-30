function map_get_coord, map, coord 
;+ purpose: convert image pixel position into solar coordinates
; for map-image
; Input: 
; map - map-struct
; coord[2,*] - coordinates of points in pixel, 
;           1d,2d,or 3d, with first dimension for axis
; Output:
; return, coord. [2,*] - with dimensions same as input
;-
 ;coord=pixel
  pcoord=coord*0.0
  sz=size(coord)
  if sz[0] eq 1 then begin
    xp=coord[0]
    yp=coord[1]
  endif $
  else if sz[0] eq 2 then begin
    xp=reform(coord[0,*])
    yp=reform(coord[1,*])
  endif $
  else if sz[0] eq 3 then begin
    xp=reform(coord[0,*,*])
    yp=reform(coord[1,*,*])
  endif
 ;convert (xp,yp) into (x,y)
 ss=size(map.data)
 x=(xp-(ss[1]-1)/2.)*map.dx + map.xc
 y=(yp-(ss[2]-1)/2.)*map.dy + map.yc

  if sz[0] eq 1 then begin
    pcoord[0]=x
    pcoord[1]=y
  endif $
  else if sz[0] eq 2 then begin
    pcoord[0,*]=x
    pcoord[1,*]=y
  endif $
  else if sz[0] eq 3 then begin
    pcoord[0,*,*]=x
    pcoord[1,*,*]=y
  endif
 return, pcoord
end

