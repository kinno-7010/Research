;+
; PROJECT:
;  SOHO-LASCO
;
;  NAME:
;       EXAMPLE_HDF_READ
;
;  PURPOSE:
;       example to show how to use TOMO_HDF_READ
;
;  CATEGORY:
;       io
;
;  CALLING SEQUENCE:
;       EXAMPLE_HDF_READ  
;         
;  SIDE EFFECTS:
;         None.
;
;  MODIFICATION HISTORY:
;         December 2015, Creation  (by J.W)
;-

pro example_hdf_read

  
  ; step 0 set global parameter
  dataset_filename = '../CR2062_1.hf5' ;
  CRstdate = '2007/10/08'       ; % carrington rotration 2062 startingdate 
 ; CRmiddate = '2007/10/12'      ; % carrington rotation 2062 middate
  CRdateofinterest = CRstdate   ; % can switch to CRmiddate here
  radiitoshow = [3,4]           ; %in Rsun  (usefull for latitude/longitude maps)
  latitudetoshow = 0            ; %in degree (usefull for radius/longitude maps)
  
  ;step 1.1 file reading 
  ;  get latitude, longitude radius and time by file reading 
  tomo_hdf_read, dataset_filename, lon, lat, rad, time, vol, misc
  
  ;step 1.2 convert date to a numeric form and get time index
  num_dateofinterest = julday(strmid(strtrim(CRdateofinterest,2),5,2), $
                              strmid(strtrim(CRdateofinterest,2),8,2), $  
                              strmid(strtrim(CRdateofinterest,2),0,4)) 
  
  num_tomostartdate = julday(strmid(strtrim(misc.startingdate,2),4,2) , $
                             strmid(strtrim(misc.startingdate,2),6,2) , $       
                             strmid(strtrim(misc.startingdate,2),0,4))

  delay_from_start = num_dateofinterest-num_tomostartdate                        
  dummy = min(abs(delay_from_start-time), time_idx)         
  

  ;step 2.1 plot latitude/longitude maps  
  ; draw latitude/longitude maps for selected radii
  for rr=0,n_elements(radiitoshow)-1 do begin
    current_radius = radiitoshow[rr]
    ;get index of current radius
    dummy = min(abs(rad-current_radius),r_idx)
    latlon_imshow1 = image(transpose(alog10(reform(vol[r_idx,*,*,time_idx])>0.001)),lon, lat, $
                           rgb_table=5,min_value=3.5, max_value=5.5,axis_style=1 ,$
                           xtitle='longitude', ytitle='latitude', dimension=[700,500], $
                           xtickvalue=[0,90,180,270,360], ytickvalue=[-90,-45,0,45,90], $
                           title='lat/lon map at '+strtrim(rad[r_idx],2)+'Rsun')
    latlon_imshow1.scale,4,4
    ;get the longitude of the observer and draw it on map
    obs_lon = misc.obscl(time_idx);
    !NULL = plot([obs_lon,obs_lon],[-90,90],/overplot)
 endfor

 ; step 2.2 plot lon/rad
 ; draw radius/longitude radius map for selected latitudes
  for ll=0,n_elements(latitudetoshow)-1 do begin
     
     current_lat = latitudetoshow[ll] ;
     dummy = min(abs(lat-current_lat), lat_idx)     
                                ;
     radlon_imshow1 = image(reverse(transpose(alog10(reform(vol[*,lat_idx,*,time_idx])>0.001)),1),lon, rad*40, $
                           rgb_table=5,min_value=3.5, max_value=5.5,axis_style=1 ,$
                            xtitle='longitude', ytitle='radius*40', dimension=[700,500], $
                            xtickvalue=[0,90,180,270,360], $
                           title='lat/lon map at '+strtrim(rad[r_idx],2)+'Rsun')
     radlon_imshow1.scale,4,4

 endfor 
  
end
