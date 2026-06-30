function map_get_pixel, map, coord
;+ purpose: convert image coordinates into pixel position
; for map-image
; Input: 
; map - map-struct
; coord[2,*] - 1d,2d,or 3d, with first dimension for axis
; Output:
; return [2,*] with dimensions same as input for pixel coord.
; of image
;-
  pcoord=coord*0.0
  sz=size(coord)
  if sz[0] eq 1 then begin
    x=coord[0]
    y=coord[1]
  endif $
  else if sz[0] eq 2 then begin
    x=reform(coord[0,*])
    y=reform(coord[1,*])
  endif $
  else if sz[0] eq 3 then begin
    x=reform(coord[0,*,*])
    y=reform(coord[1,*,*])
  endif
 ;convert (x,y) into (xp,yp)
 ss=size(map.data)
 xp=(ss[1]-1)/2.+(x-map.xc)/map.dx
 yp=(ss[2]-1)/2.+(y-map.yc)/map.dy

  if sz[0] eq 1 then begin
    pcoord[0]=xp
    pcoord[1]=yp
  endif $
  else if sz[0] eq 2 then begin
    pcoord[0,*]=xp
    pcoord[1,*]=yp
  endif $
  else if sz[0] eq 3 then begin
    pcoord[0,*,*]=xp
    pcoord[1,*,*]=yp
  endif
 return, pcoord
end

