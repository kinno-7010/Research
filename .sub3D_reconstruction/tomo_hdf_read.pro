;+
; PROJECT:
;  SOHO-LASCO
;
;  NAME:
;       TOMO_HDF_READ
;
;  PURPOSE:
;       read tomography result store in .hdf (or .hf5) file, 
;
;  CATEGORY:
;       io
;
;  CALLING SEQUENCE:
;         TOMO_HDF_READ, i_hdf_file, o_lon, o_lat, o_rad, o_time, o_vol, o_misc 
;         
;  INPUTS :
;         i_hdf_file: input tomo result file store in .hdf (or .hf5 format)
;
;  OUTPUTS:
;        o_lon: longitude vector
;        o_lat: latitude vector
;        o_rad: radius vector 
;        o_time: time vector
;        o_vol: 4D tomographic volume (3D + t, coords are t,
;        theta(lat), phi(lon), t(time)
;        o_misc: structure that contains starting date format yyyymmdd  
;                                        ending date format yyyymmdd
;                                        observer longitude (vector of
;                                        longitude of the same size as
;                                        the t coordinate)
;  SIDE EFFECTS:
;         None.
;
;  MODIFICATION HISTORY:
;         December 2015, Creation  (by J.W)
;-

pro tomo_hdf_read, i_idf_file, o_lon, o_lat, o_rad, o_time, o_vol, o_misc 
  
  ; sanity check
  if (not(file_test(i_idf_file))) then $
     message, i_idf_file, 'not_found'
  
  ; Open the hdf5 file
  file_id = h5f_open(i_idf_file)
  
  ; Open the latitude dataset within the file. 
  ; This is located within /axes  group.
  latitudes_id = h5d_open(file_id, '/axes/latitudes')
  ; Read in the actual latitudes data
  o_lat = h5d_read(latitudes_id)
  ; Close the latitude dataset within the file.
  h5d_close, latitudes_id

  ; Open the longitude dataset within the file
  ; This is located within /axes group.
  longitudes_id = h5d_open(file_id, '/axes/longitudes')
  o_lon = h5d_read(longitudes_id)
  ; Close the longitude dataset within the file.
  h5d_close, longitudes_id

  ; Open the radii dataset within the file
  ; This is located within /axes group
  radius_id = h5d_open(file_id, '/axes/rad')
  o_rad = h5d_read(radius_id)
  ; Close the longitude dataset within the file 
  h5d_close, radius_id

  ; Open the time dataset within the file
  ; This is locatted within /axes group
  time_id =  h5d_open(file_id, '/axes/time')
  o_time = h5d_read(time_id)
  ; Close the time dataset within the file
  h5d_close, time_id

  ; Open the 4d volume dataset within the file
  ; This is located within /volume group
  vol_id = h5d_open(file_id, '/volume/dataset_4D')
  o_vol = h5d_read(time_id)
  ; Close the 4d volume dataset within the file
  h5d_close, vol_id

  ; prepare the o_misc structure
  o_misc = {name:'', obscl:fltarr(n_elements(o_time)), startingdate:'yyyymmdd', endingdate:'yyyymmdd'}

  ; Open the observer carrington longitude dataset within the file
  ; This is located within /misc group
  obscl_id = h5d_open(file_id, '/misc/obscl')
  o_misc.obscl = h5d_read(obscl_id)
  ; Close the observer carrington longitude dataset within the file
  h5d_close, obscl_id

  ; Open the starting date dataset within the file
  ; This is located within /misc group
  startdate_id = h5d_open(file_id, '/misc/startingdate')
  o_misc.startingdate = h5d_read(obscl_id)
  ; Close the observer starting date dataset within the file
  h5d_close, startdate_id 
   
  ; Open the ending date dataset within the file
  ; This is located within /misc group
  enddate_id = h5d_open(file_id, '/misc/endingdate')
  o_misc.endingdate = h5d_read(enddate_id)
  ; Close the observer starting date dataset within the file
  h5d_close, enddate_id

  ; Close the hdf5 file
  h5f_close, file_id
  
  
end
