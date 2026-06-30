pro gen_para
; generate the list for input file names
 f='pb_cor1A_sz128_'
 num=28
 openw,1,'data-list.txt'
 for i=1,num do begin
   cnm=	 strcompress(string(i, format='(i3.3)'),/remove_all)
   printf,1,f+cnm+'.dat'
  endfor
 close,1
 print,'file [data-list.txt] is created!'
end

