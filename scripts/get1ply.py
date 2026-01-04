def get1ply(filename):
    from plyfile import PlyData
    import os
    import numpy as np
    #没找到文件会报错
    plydata = PlyData.read(filename)

    vertex =  plydata ['vertex']
       
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']


    combined = np.stack((x, y, z), axis=-1)
    return combined
