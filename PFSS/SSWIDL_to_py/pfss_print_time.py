"""
====================================================================

pfss_print_time.py - in a loop, prints time remaining/elapsed to finish loop

usage:  pfss_print_time(str_prefix, it, nit, tst, slen, elapsed=False)
        where str_prefix = string prepended to time left message
              it = current iteration number (starting from 1)
              nit = total number of iterations
              tst = time.time() just before entering loop
              slen = on input, contains string length of message to erase,
                     on output, contains length of the message printed here
              elapsed = set flag if you want time elapsed to be printed

M.DeRosa - 11 Mar 2000 - created (IDL version)
         - 15 Mar 2000 - now prints time left in seconds if under 1
                         minute (for the truly impatient!)
         -  1 Aug 2000 - added capability for tst to be calculated
                         upon entering routine if not already done 
         -  8 Feb 2001 - added extra print statement if on final iteration
         - 10 Jan 2002 - changed routine to be able to print both time 
                         remaining and time elapsed, renamed as print_time
                         (formerly print_time_left)
         - 23 Jan 2006 - now does a CR without backspacing by using the
                         CR ASCII character with a $ format, instead of
                         using the BS (backspace) character.  This works
                         better in the IDLDE and doesn't TTY behavior
Converted to Python - 2025

====================================================================
"""

import time
import sys
from typing import Optional, List


def pfss_print_time(str_prefix: str, it: int, nit: int, 
                   tst: Optional[float] = None, 
                   slen: Optional[List[int]] = None, 
                   elapsed: bool = False) -> List[int]:
    """
    Prints time remaining/elapsed to finish loop.
    
    Parameters:
    -----------
    str_prefix : str
        String prepended to time left message
    it : int
        Current iteration number (starting from 1)
    nit : int
        Total number of iterations
    tst : float, optional
        time.time() just before entering loop (auto-calculated if None)
    slen : list, optional
        Contains string length of message to erase (modified in-place)
    elapsed : bool, optional
        If True, print time elapsed instead of time remaining
        
    Returns:
    --------
    list
        Updated slen value
    """
    
    # Initialize slen if not provided
    if slen is None:
        slen = [0]
    
    # Set tst if not set or if first iteration
    if tst is None or it == 1:
        tst = time.time()
    
    # Calculate time remaining in minutes (or seconds if under 1 minute)
    telapsed = time.time() - tst  # time elapsed (seconds)
    
    if it > 1:  # avoids division by zero errors
        if elapsed:
            time_val = telapsed
        else:
            itrate = telapsed / (it - 1)  # current iteration rate (iterations per second)
            time_val = (nit - it + 1) * itrate  # time remaining (seconds)
        
        if time_val < 59.5:
            time_val = round(time_val)
            label = ' second'
        else:
            time_val = round(time_val / 60.0)
            label = ' minute'
        
        timestr = str(int(time_val)) + label
        if time_val != 1:
            timestr = timestr + 's'
    else:
        timestr = '---'
    
    if elapsed:
        timestr = 'elapsed = ' + timestr
    else:
        timestr = 'left = ' + timestr
    
    # Erase previous message
    if slen[0] > 0:
        sys.stdout.write('\r')
        sys.stdout.flush()
    
    # Print time left
    outstr = (str_prefix + 'on iteration ' + str(it) + ' of ' + str(nit) + 
              ', time ' + timestr + '     ')
    sys.stdout.write(outstr)
    sys.stdout.flush()
    
    # Set slen variable
    slen[0] = len(outstr)
    
    # Print newline if on final iteration
    if it == nit:
        print()
    
    return slen