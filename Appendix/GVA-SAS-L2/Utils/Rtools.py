import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import numpy as np

def rds2dict(filename):
    ''' 
    Creates a Python Dictionary with Numpy Array elements 
    from an R .RDS file
    '''
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']
    rds = readRDS(filename)
    
    data = {} 

    for i in range(len(rds.names)):
        data[rds.names[i]] = np.array(rds[i])
       
    # ensures scalars are represented that way (not ndarray)    
    for item in data:
        if data[item].shape ==(1,) or data[item].shape ==(1,1):
            data[item] = data[item][0]
     
    return data



