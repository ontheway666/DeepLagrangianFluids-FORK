#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import json
import time
import hashlib
import util
import getpartEmit
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles
import open3d as o3d
from write_ply import write_ply
np.random.seed(5678)
from prms import *





# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         print('[MEMORY GROWTH]')
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)



boarddis=0
startstep=0
tempdone=False


assert(not prm_train)


if(prm_continuerun):
    startstep=prm_resumestep-1
    if(prm_simrigidwithoutfluid):
        startstep=0


# xx=tf.random.normal((1,1))
# print('\n\n\n[run]\n\n\n')
# print(xx)






export_num=0
scenejsonname="error"





def hashm(velocities):
    tt = tuple(tuple(row) for row in velocities) 
    matrix_str = str(tt)  
    # 使用哈希函数计算哈希值  
    hashh = hashlib.md5(matrix_str.encode()).hexdigest()   
    return hashh

def write_particles(path_without_ext, pos, vel=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}



    if not vel is None:
        arrs['vel'] = vel

        
    if(prm_savelvel):
        arrs={'vel':vel}
        np.savez(path_without_ext + '.npz', **arrs)
        np.save(path_without_ext)
        print('zxc save vel')

    if options and options.write_ply:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
        o3d.io.write_point_cloud(path_without_ext + '.ply', pcd)
        # assert(False)

    if options and options.write_bgeo:
        write_bgeo_from_numpy(path_without_ext + '.bgeo', pos, vel)

#zxc 验证集
def run_sim_tf(trainscript_module, weights_path, scene, num_steps, output_dir,
               options):

    # init the network
    global export_num


    stm=time.time()
    model = trainscript_module.create_model()
    model.init()
    model.load_weights(weights_path, by_name=True)
    # api 


    
    #know
    if(prm_mix):
        print('[mix]')



        model2 = trainscript_module.create_model()
        model2.init()
        model2.load_weights("csm_mp300.h5", by_name=True)
        #注意加载参数，却不会加载dt和gravity。它是在create_model时确定的

        model3 = trainscript_module.create_model()
        model3.init()
        model3.load_weights("pretrained_model_weights.h5", by_name=True)

        #prm_

        model4 = trainscript_module.create_model()
        model4.init()
        model4.load_weights("csm300_1111.h5", by_name=True)


        model5 = trainscript_module.create_model()
        model5.init()
        model5.load_weights("stabledf.h5", by_name=True)



        # layername=['cvf','cvo','cv1','cv2','cv3','d0','d1','d2','d3']
        # mname=['csm_df300_1111','csm_mp300','pretrained_model_weights','csm300_1111']
        # print('[layer TF]')
        # print(model.summary())
        # print('[trainable]')
        # print(len(model.trainable_variables))#18
        # for i in model.trainable_variables:#list
        #     print(i.shape)
        #     print(i.name)
        #     print(type(i))
        # assert(False)
        

        # for idx,m in enumerate([model,model2,model3,model4]):
        #     layerid=0
        #     for layer in tqdm(m.layers):
        #         print(str(layerid)+'\t[one layer]---------------')
        #         # print(layer)
        #         x=layer.get_weights()
        #         print(len(x))
        #         # print(type(x))

        #         # print(type(x[0]))
        #         print(x[0].shape)
        #         print(x[1].shape)
        #         np.savez('/w/cconv-dataset/npweight/'+mname[idx]+'/'+layername[layerid]+'.npz', weights=x[0], biases=x[1])
        #         layerid+=1
        # exit(0)

    else:
        print('[single]')
        


    #COPY
    etm=time.time()
    print('[models loading time](s)\t'+str(etm-stm))


    print('./cache/'+scenejsonname+'-f.npy')
    if not os.path.exists('./cache/'+scenejsonname+'-box.npy'):#know
        print('[no scene ply cache]') 

        # prepare static particles
        walls = []
        wallinfo=[]


        print('-------wall------')
        for x in scene['walls']:
            print('sampling \t'+str(x['path']))
            if(not 'scale' in x):
                points, normals=obj_surface_to_particles(x['path'])
            else:
                points, normals = obj_surface_to_particles(x['path'],scalefactor=x['scale'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
                print('[invert normal]')
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))

            print('wall sp\t'+str(points.shape))
            wallinfo.append(points.shape)
        box = np.concatenate([x[0] for x in walls], axis=0)
        print('all wall sp\t'+str(box.shape))
        # assert(False)

        box_normals = np.concatenate([x[1] for x in walls], axis=0)

        # export static particles
        write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

        # compute lowest point for removing out of bounds particles
        min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))

        # prepare fluids
        fluids = []


        if(not prm_myemit):
            for x in scene['fluids']:
                points = obj_volume_to_particles(x['path'])[0]
                points += np.asarray([x['translation']], dtype=np.float32)
                velocities = np.empty_like(points)
                velocities[:, 0] = x['velocity'][0]
                velocities[:, 1] = x['velocity'][1]
                velocities[:, 2] = x['velocity'][2]
                range_ = range(x['start'], x['stop'], x['step'])
                fluids.append((points, velocities, range_))
                #zxc
                np.save('./cache/'+scenejsonname+"-f",fluids)
        np.save('./cache/'+scenejsonname+"-box",box)
        np.save('./cache/'+scenejsonname+"-boxn",box_normals)
        np.save('./cache/'+scenejsonname+"-wallinfo",wallinfo)

    else:
        print('[use cache]')
        # assert(False)
        if(not prm_myemit):
            fluids=     np.load('./cache/'+scenejsonname+"-f.npy",allow_pickle=True)
        box=        np.load('./cache/'+scenejsonname+"-box.npy",allow_pickle=True)
        box_normals=np.load('./cache/'+scenejsonname+"-boxn.npy",allow_pickle=True)
        wallinfo=   np.load('./cache/'+scenejsonname+"-wallinfo.npy",allow_pickle=True)
        print(wallinfo)
        # assert(False)

        #test 2.5m
        # print(box.shape)
        # print(fluids.shape)#1 3
        # print(fluids[0,0])
        # print(fluids[0,1])
        # print(fluids[0,2])
        # random_array = np.random.rand(300000, 3)
        # #这里导入30w不行

        # # random_array = np.random.rand(600000, 3)


        
        # fluids[0,0]=random_array
        # del random_array
        # fluids[0,1]=np.zeros_like(fluids[0,0])



    
    if(prm_myemit):

        from getlattice import generate_cube_points

        # myemit
        # points_emit=generate_cube_points(
        #     center=[-16.4,-3.4,0.0],
        #     length=0,
        #     height=5.44,
        #     width=1.25,
        #     spacing=0.05
        # )
        #slope
        # points_emit=generate_cube_points(
        #     center=[-16.4,-3.4+2,0.0],
        #     length=0,
        #     height=5.44,
        #     width=1.25,
        #     spacing=0.05
        # )


        # slope move 
        # points_emit=generate_cube_points(
        #     center=[-16.4-2.3,-3.4+2,0.0],
        #     length=0,
        #     height=5.44,
        #     width=1.25,
        #     spacing=0.05
        # )


        # slope movex+
        # points_emit=generate_cube_points(
        #     center=[-16.4-4,-3.4+2,0.0],
        #     length=0,
        #     height=5.44,
        #     width=1.25,
        #     spacing=0.05
        # )
        # vel_emit=np.zeros_like(points_emit)
        # vel_emit[:,0]=4.5
        # vel_emit[:,1]=-1.5


        # vel_emit[:,0]=3.125
        # vel_emit[:,1]=-1.2


        # vel_emit[:,0]=4
        # vel_emit[:,1]=-1.5


        range_=range(0, 170, 1)
        #prm_



    
        # --------------test max num- 测试最大承载粒子数----------------------------------------
        range_=range(0, 1, 1)

        #90w board
        # points_emit=generate_cube_points(
        #     center=[-16.4-4,-3.4+2,0.0],
        #     length=0,
        #     height=38.44,
        #     width=60.25,
        #     spacing=0.05
        # )
        if(scenejsonname=='maxpartnum.json'):
            #90w block, 187w
            points_emit=generate_cube_points(
                    center=[0,-3,0],
                    length=2.5,
                    height=2.5,
                    width=18,
                    spacing=0.05
                )
            vel_emit=np.zeros_like(points_emit)
            print(points_emit.dtype)#float32





        # center=[0,-0.5,0],
        # high fluid 这一组数据不转float64
        # 80w
        # points_emit=generate_cube_points(
        #     center=[-5,-0.5,-5],
        #     length=2.5,
        #     height=2.5,
        #     width=16,
        #     spacing=0.05
        # )

        # global scenejsonname



        if(scenejsonname=='high_fluid_mcvsph-fluid.json'):
            # 50w highFluid
            center=[0,-0.5,0]
            # high fluid 这一组数据不转float64
            points_emit=generate_cube_points(
                center=[-5,-3,-5],
                length=2.5,
                height=2.5,
                width=10,
                spacing=0.05
            )
            box[:,0]-=5
            box[:,2]-=5
            points_emit=points_emit.astype(np.float16)
            vel_emit=np.zeros_like(points_emit)

     



        # points_emit=np.load("./tempInitScene-70w.npz")['mat1']


        #这个数据不能直接送入，内存会炸，必须赋值到一个numpy数组中
        # data=np.load("./tempInitScene-70w.npy")
        # points_emit=data
        # print(points_emit.dtype)#float64



        # # points_emit=np.load("./tempInitScene-40w.npy")
        # vel_emit=0
        # tempcnt=0
        # maxv=0
        # minv=0
        # print(np.max(data[:,0]))
        # print(np.max(data[:,1]))
        # print(np.max(data[:,2]))
        # print(np.min(data[:,0]))
        # print(np.min(data[:,1]))
        # print(np.min(data[:,2]))
        # # points_emit=np.zeros_like(points_emit)
        # for index, value in np.ndenumerate(data):
        #     tempcnt+=1
        #     # if(tempcnt<100):
        #     #     print(index)
        #     dim0=index[0]
        #     dim1=index[1]
        #     points_emit[dim0,dim1]=data[dim0,dim1]
        # print(tempcnt)
        # vel_emit=np.zeros_like(points_emit)


        # from get1ply import get1ply
        # points_emit=get1ply("./tempInitScene0000.ply")



        elif(scenejsonname=='rotating_panel.json'):
            scale=2  #与scale=1的box配合使用

            scale*=1.5

            points_emit,vel_emit, range_ =getpartEmit.get_partemit_rotatingpanel(scale=scale)


        elif(scenejsonname=='example_static1.5x.json'):
            scale=1.5
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_examplestatic(scale=scale)

        elif(scenejsonname=='streammultiobjs.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_streammultiobjs()

        elif(scenejsonname=='streammultiobjs2x.json' or scenejsonname=='streammultiobjs2xCut.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_streammultiobjs(scale=2)
        elif(scenejsonname=='streammultiobjs3x.json' or \
             scenejsonname=='streammultiobjs3xcut.json'or \
             scenejsonname=='streammultiobjs3xcutbunny12.json' or \
             scenejsonname=='streammultiobjs3xcutbunny12-1.5x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_streammultiobjs(scale=3)
        elif(scenejsonname=='streammultiobjsHorizon.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_streammultiobjsHorizon(scale=2)
  

        elif(scenejsonname=='watervessel.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=2)

        elif(scenejsonname=='watervessel1.13x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=2.26)

        elif(scenejsonname=='watervessel1.4x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=2.8)
        elif(scenejsonname=='watervessel1.5x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=3.0)
            # print(points_emit.shape)
            # assert(False)


        elif(scenejsonname=='watervessel1.06x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=2.12)


        elif(scenejsonname=='watervessel0.7x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=1.4)

        elif(scenejsonname=='watervessel0.8x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=1.6)


        elif(scenejsonname=='watervessel0.9x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_watervessel(scale=1.8)


        elif(scenejsonname=='wavetower.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_wavetower(scale=2)

        elif(scenejsonname=='momeConse.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_momeConse()

            box=box[:5,:]
            box_normals=box_normals[:5,:]
            print('box sp zxc-------')
            print(box.shape)

        elif(scenejsonname=='wavetowerstatic.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_wavetowerstatic()

        elif(scenejsonname=='wavetowerstatic1.3x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_wavetowerstatic(scale=1.3)

        elif(scenejsonname=='propeller.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_propeller()

        elif(scenejsonname=='propellerlarge.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_propellerlarge()

        elif(scenejsonname=='propellerlarge1.5x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_propellerlarge(scale=1.5)

        elif(scenejsonname=='propeller2large.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_propellerlarge()
        

        elif(scenejsonname=='propeller2large2.json' or\
           scenejsonname=='propeller2+large2.json'or\
           scenejsonname=='propeller2+large2fill.json' ):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_propellerlarge2()

        elif(scenejsonname=='propeller2large21.5x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_propellerlarge2(scale=1.5)


        elif(scenejsonname=='propellerbiglarge2.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_propellerbiglarge2()


        elif(scenejsonname=='taylorvortex.json' or scenejsonname=='rotatingpanelstatic.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_taylorvortex()


        elif(scenejsonname=='rotatingpanelstatic2x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_taylorvortex(scale=2.0)           


        elif(scenejsonname=='taylorvortex1.5scale.json' or scenejsonname=='rotatingpanelstatic1.5x.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_taylorvortex(scale=1.5)


        elif(scenejsonname=='rotatingpanelstatic2x_thin.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_taylorvortex_thin(scale=2)


        elif(scenejsonname=='mc_ball_2velx_0602_wake.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_0602_wake(scale=1)

        elif(scenejsonname=='mc_ball_2velx_0602_wake_wide.json'):
            points_emit,vel_emit, range_ =getpartEmit.get_partemit_0602_wake_wide(scale=1)

        else:
            print(scenejsonname)
            print('no match scene,maybe set myemit=0')
            assert(False)
     

        print(points_emit.shape)

        # points_emit=points_emit.astype(np.float64)
        print(points_emit.dtype)


        print(points_emit[5:,:])
        # import sys
        # from pympler import asizeof
        print(sys.getsizeof(points_emit))
        # assert(False)
        # import os
        import resource
        
        # 获取当前进程的PID
        pid = os.getpid()
        
        # 获取当前进程的内存使用情况（以KB为单位）
        memory_usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # print(f"当前Python程序占用内存: {memory_usage_kb} KB")
        # print(asizeof.asizeof(points_emit))

        # assert(False)
        
        # assert(False)
       

        # points_emit=points_emit[:700000,:]
        # points_emit2=points_emit.copy()
        # del points_emit
        # points_emit=points_emit2
        #直接截断700000不够用
        #加上边界：800000 OK
        

        print('[custom init fluid sp]')
        # print(points_emit.shape)


        # box=box[:1,:]
        # box_normals=box_normals[:1,:]
        # test max num---------------------------------------------------------------------------






        



        fluids=[]
        for x in scene['fluids']:
            # points = obj_volume_to_particles(x['path'])[0]
            # points += np.asarray([x['translation']], dtype=np.float32)
            # velocities = np.empty_like(points)
            # velocities[:, 0] = x['velocity'][0]
            # velocities[:, 1] = x['velocity'][1]
            # velocities[:, 2] = x['velocity'][2]

            
            fluids.append((points_emit,vel_emit,range_))
            # assert(False)
            # print('this method')

                #zxc

    max_y = np.max(box[:, 1]) 
    min_y = np.min(box[:, 1]) #- 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))
    max_z = np.max(box[:, 2])
    min_z = np.min(box[:, 2])

    max_x = np.max(box[:, 0])
    min_x = np.min(box[:, 0])


    # box=box[:1,:]

    if(prmcutbox):
        print('CUT BOX')
        if(scenejsonname=="wavetowerstatic.json"):
            mask=box[:wallinfo[0][0]:,1] < 2
     
        elif(scenejsonname=="propeller.json"):
            mask=box[:wallinfo[0][0]:,1] < 0.97
        elif(scenejsonname=="streammultiobjs3x.json" or \
             scenejsonname=="streammultiobjs3xcut.json" or \
             scenejsonname=="streammultiobjs3xcutbunny12.json" or \
             scenejsonname=="streammultiobjs3xcutbunny12-1.5x.json"):
            mask=box[:wallinfo[0][0]:,1] < 4.5

        elif(scenejsonname=="watervessel1.13x.json"):
            mask=box[:wallinfo[0][0]:,1] < 1.5066
            

        elif(scenejsonname=="propellerbiglarge2.json"):
            mask=box[:wallinfo[0][0]:,1] < 2
        
        elif(scenejsonname=="propeller2large21.5x.json"):
            mask=box[:wallinfo[0][0]:,1] < 1.5

        elif(scenejsonname=="propellerlarge1.5x.json"):
            mask=box[:wallinfo[0][0]:,1] < 1

        elif(scenejsonname=="mc_ball_2velx_0602_wake.json" 
        or scenejsonname=="mc_ball_2velx_0602_wake_wide.json"):
            mask=box[:wallinfo[0][0]:,1] < -2.5
        
        else:
            print('no cutbox ways for scene:' + str(scenejsonname))
            assert(False)
            

        filterednum=np.sum(mask.astype(int))
        print(mask.shape)#partnum 1
        print('after filter:\t'+str(filterednum))
        box0=box                [:wallinfo[0][0],:]
        box_normals0=box_normals[:wallinfo[0][0],:]
        box0=                box0[mask]
        box_normals0=box_normals0[mask]
        box=        np.concatenate([box0,        box        [wallinfo[0][0]:,:]])
        box_normals=np.concatenate([box_normals0,box_normals[wallinfo[0][0]:,:]])

        print(box.shape)
        print(box_normals.shape)

      
        wallinfo[0]=[filterednum,3]

        
    if(scenejsonname=="propeller2+large2.json" or\
       scenejsonname=="propeller2+large2fill.json"):
        print('CUT PROP')
        
        rig0=wallinfo[0][0]
        rig1=wallinfo[1][0]
        rig2=wallinfo[2][0]
        mask= np.linalg.norm(box[rig0:rig0+rig1,[0,2]], axis=1)<0.6   #<0.5

       

        mask3=box[rig0:rig0+rig1,1]>0.76

        mask=np.logical_or(mask,mask3)


        filterednum=np.sum(mask.astype(int))
        # assert filterednum>0
        box1=box                [rig0:rig0+rig1,:]
        box_normals1=box_normals[rig0:rig0+rig1,:]
        box1=                box1[np.logical_not(mask) ]
        box_normals1=box_normals1[np.logical_not(mask) ]
        box=        np.concatenate([box        [:rig0,:],            box1,       box [rig0+rig1:,:]])
        box_normals=np.concatenate([box_normals[:rig0,:],    box_normals1,box_normals[rig0+rig1:,:]])
        
        wallinfo[1]=[rig1-filterednum,3]   
        print(wallinfo)         


        # assert(False)
        # mask=mask[:,0]
        
        # box[:wallinfo[0][0]:,:] = 
    # box_normals=box_normals[wallinfo[0][0]:,:]

    # mask=box[:,1] < -10
    # box=box[mask]
    # box_normals=box_normals[mask]
    # box=np.array([0.0, 0.0, 0.0])
    # box_normals=np.array([1.0, 0.0, 0.0])
    # box=util.ensure_2d(box).T
    # box_normals=util.ensure_2d(box_normals).T
    # min_x=0
    # min_y=0
    # min_z=0
    # max_x=6
    # max_y=6
    # max_z=6
    



    print('[box sp]')
    print(box.shape)
    if(prm_myemit):
        print('[total num]')
        print(box.shape[0]+points_emit.shape[0])
    print(min_y)
    



    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)
    if(prm_continuerun):
        print('[CONTINUE RUN]')
        from get1ply import get1ply
        pos1= get1ply(os.path.join(output_dir,'fluid_{0:04d}.ply'.format(prm_resumestep-2)))
        pos2=get1ply(os.path.join(output_dir,'fluid_{0:04d}.ply'.format(prm_resumestep-1)))
        vel=(pos2-pos1)/0.016
        pos=pos2
        print(pos.shape)
        print(vel.shape)
        print(np.min(pos[:,0]))
        print(np.min(pos[:,1]))
        print(np.min(pos[:,2]))
        print(np.max(pos[:,0]))
        print(np.max(pos[:,1]))
        print(np.max(pos[:,2]))
        print(np.mean(vel))




    starttime=time.time()
    if(prm_round):
        points=    np.round(points,    eps)
        velocities=np.round(velocities,eps)

    if(not prm_continuerun):
        write_ply(
                    path=output_dir+'/box',
                    frame_num=1,
                    dim=3,
                    num=box.shape[0],
                    pos=box)

    for step in tqdm(range(startstep,num_steps)):
        # print('[num_steps]')
        # print(num_steps)
        # time.sleep(3000)
        # add from fluids to pos vel arrays
        # import tensorflow as tf
        # tf.reset_default_graph()
        # tf.keras.backend.clear_session()
        for points, velocities, range_ in fluids:
        # for    points_emit,vel_emit,range_ in fluids:
            if (step in range_ and not prm_continuerun):  # check if we have to add the fluid at this point in time
                
                print(type(pos))#eagerTensor
                print('------tp pos---------')
                pos = np.concatenate([pos, points], axis=0)#know
                # del fluids
                # del points_emit
                # del vel_emit
                # del points
                # import gc
                # gc.collect()
                # print(type(pos))
                vel = np.concatenate([vel, velocities], axis=0)
                # del vel
                
                
                # import tensorflow as tf
                # vel = tf.zeros_like(pos)
                # vel=tf.cast(vel,tf.float32)
        # 这里才是增长粒子的一般方式


        #wrong
        # if(step==2):
        #     pos=np.random.rand(500000, 3)
        #     vel=np.zeros_like(pos)


        if(prmrestrict_upward and not(prm_continuerun and step<prm_resumestep) ):
            if(not isinstance(pos, np.ndarray)):
                vel=vel.numpy()

            #case1
            # mask=np.logical_and( vel[:,1]>0.5,  pos[:,1]>0.5)
            
            #case2
            mask=np.logical_and( vel[:,1]>0,  pos[:,1]>0.1)#朝上运动的粒子
            mask=np.logical_and( np.sqrt(np.sum(vel**2,axis=1))>1.0,mask)#速度过大，白色


            # vel[mask,1]*=np.exp(-0.1 * np.abs(vel[mask,1]))
            vel[mask,1]*=0.5
            
            
            print('[upward restrict]\t'+str(np.sum(mask.astype(np.int))))
            
            import tensorflow as tf
            vel=tf.convert_to_tensor(vel)
            vel=tf.cast(vel,tf.float32)


        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            # if(prm_exportgap!=1):
            #     fluid_output_path = os.path.join(output_dir,
            #                         'fluid_{0:04d}'.format(export_num))
                

            if(prm_outputInitScene):
                if(step!=0):
                    exit(0)
                print('[outputInitScne]')
                print(box.shape)
                print(pos.shape)
                write_ply(
                    path="./temp/box-",
                    frame_num=1,

                    dim=3,
                    num=box.shape[0],
                    pos=box)
                # write_ply(
                #     path="./temp/boxn-",
                #     frame_num=1,

                #     dim=3,
                #     num=box_normals.shape[0],
                #     pos=box_normals)
                # write_ply(
                #     path="./temp/fluid-pos",
                #     frame_num=1,
                #     dim=3,
                #     num=pos.shape[0],
                #     pos=pos)
                # write_ply(
                #     path="./temp/fluid-vel",
                #     frame_num=1,

                #     dim=3,
                #     num=vel.shape[0],
                #     pos=vel)

                # np.save("./sp/Box",box)
                # np.save("./sp/POS",pos)
                # np.save("./sp/VEL",vel)


            if( (step%prm_exportgap==0 or (step%prm_exportgap==1 and prm_exportgap>1)) and\
            (not( prm_continuerun and step==prm_resumestep-1)) and\
            (not( prm_continuerun and step<prm_resumestep))
            ):
                if isinstance(pos, np.ndarray):
                    # 先记录数据，再推理
                    
                    write_particles(fluid_output_path, pos, vel, options)
                    export_num+=1
                else:
                    # prm_
                    # from write_ply import write_plyIdx
                    # write_plyIdx(path=fluid_output_path,
                    # frame_num=step,
                    # num=pos.shape[0],
                    # pos=pos,
                    # attr=model.correctmodel_pointwise)

                    write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                    options)
                    export_num+=1
                    

            #检查网络随机性
            # pos=np.random.rand(*pos.shape)
            # vel=np.random.rand(*vel.shape)
            # box=np.random.rand(*box.shape)
            # box_normals=np.random.rand(*box_normals.shape)


            #强制在每次推理都使用tensor
            if(isinstance(pos,np.ndarray)):
                pass
                # import tensorflow as tf
                # post=tf.convert_to_tensor(pos)
                # velt=tf.convert_to_tensor(vel)
                # post=tf.cast(post,tf.float32)
                # velt=tf.cast(velt,tf.float32)

                # del pos
                # del vel

                # inputs = (post, velt, None, box, box_normals)
                inputs = (pos, vel, None, box, box_normals)



            else:
                # import tensorflow as tf
                # global tempdone
                # if(tempdone==False):
                #     pos=tf.random.uniform(shape=[500000, 3], minval=0, maxval=1, dtype=tf.float32) 
                #     vel=tf.zeros_like(pos)
                #     tempdone=True
                inputs = (pos, vel, None, box, box_normals)

            import tensorflow as tf
            # global scenejsonname
            if(prm_wallmove):

              
                    
                
                #规定：需要移动的wall恰好是第1个
                from movewall_strategy import movewall_still,rotationself,boatdown,moverigid
                global movetime

                wallmoveidx=wallinfo[0][0]
                
                if(prm_motion=='0602'):     
                    pass
                elif(prm_motion=='still'):
                    movewall_still(step=step,wallmoveidx=wallmoveidx,box=box)
                elif(scenejsonname=='example_propeller.json'):
                    assert(False)
                    rotationself(wallmoveidx=wallmoveidx,\
                                box=box)
                elif(scenejsonname=='rotating_panel.json'):
                    scale=1.5
                    if(step<120):
                        pass
                    elif(step<=1000):
                        # print(np.mean(box[wallmoveidx:,0]))
                        # print(np.mean(box[wallmoveidx:,1]))
                        # print(np.mean(box[wallmoveidx:,2]))
                        # assert(False)
                        rotationself(wallmoveidx=wallmoveidx,\
                                    box=box,center=tf.constant([3.6,2.0,3.5])*scale,
                                    vel=1.0/120.0)
                    elif(step<=1500):
                        box[wallmoveidx:,1]+=0.1*scale

                elif(scenejsonname=='watervessel.json' or \
                scenejsonname=='watervessel1.5x.json' or \
                scenejsonname=='watervessel1.13x.json' or \
                scenejsonname=='watervessel1.06x.json' or \
                scenejsonname=='watervessel0.8x.json' or \
                scenejsonname=='watervessel0.7x.json' or \
                scenejsonname=='watervessel0.9x.json' or \
                scenejsonname=='watervessel1.4x.json' or \
                scenejsonname=='watervessel1.5x.json'
                ):
                    
                    scale=2.0

                    if(scenejsonname=='watervessel1.4x.json'):
                        scale*=1.4

                    if(scenejsonname=='watervessel1.5x.json'):
                        scale*=1.5


                    if(scenejsonname=='watervessel1.13x.json'):
                        scale*=1.13
            

                    if(scenejsonname=='watervessel1.06x.json'):
                        scale*=1.06


                    if(scenejsonname=='watervessel0.8x.json'):
                        scale*=0.8


                    if(scenejsonname=='watervessel0.7x.json'):
                        scale*=0.7


                    if(scenejsonname=='watervessel0.9x.json'):
                        scale*=0.9


                    

                    print(np.mean(pos[:,0]))
                    print(np.mean(pos[:,1]))
                    print(np.mean(pos[:,2]))
                    # assert(False)


                    waittime=100
                    downtime=40
                    tempomega=-1.0/40.0


                    # test, util water become static
                    waittime=200



                    #df
                    # tempomega=-0.6/40.0


                    # #MT   MP
                    # tempomega=-0.3/40.0


                    if(scenejsonname=='watervessel1.13x.json'):
                        tempomega=-0.6/40.0     #ie -2.7 deg

                        
                        # test
                        # tempomega=-0.3/40.0


                    if(scenejsonname=='watervessel1.06x.json' or \
                       scenejsonname=='watervessel0.8x.json' or \
                       scenejsonname=='watervessel0.7x.json' or \
                       scenejsonname=='watervessel0.9x.json' or \
                       scenejsonname=='watervessel1.4x.json' or \
                       scenejsonname=='watervessel1.5x.json'):
                        tempomega=-0.6/40.0    



                    if(step<=waittime):
                        pass
                    elif(step>waittime and step<=downtime+waittime):
                        boatdown(wallmoveidx,box)
                    else:
                        if(scenejsonname=='watervessel1.4x.json'):
                            rotationself(wallmoveidx,box,center=tf.constant([8.379,0.0,8.4]),vel=tempomega)
                        if(scenejsonname=='watervessel1.5x.json'):
                            rotationself(wallmoveidx,box,center=tf.constant([8.978,0.0,9.0]),vel=tempomega)


                        if(scenejsonname=='watervessel1.13x.json'):
                            rotationself(wallmoveidx,box,center=tf.constant([6.76342,0.0,6.77999]),vel=tempomega)
                        


                        elif(scenejsonname=='watervessel1.06x.json'):
                            rotationself(wallmoveidx,box,center=tf.constant([6.36,0.0,6.36]),vel=tempomega)

                        elif(scenejsonname=='watervessel0.8x.json'):
                            rotationself(wallmoveidx,box,center=tf.constant([2.4,0.0,2.4])*scale,vel=tempomega)


                        elif(scenejsonname=='watervessel0.7x.json'):
                            rotationself(wallmoveidx,box,center=tf.constant([2.1,0.0,2.1])*scale,vel=tempomega)
                        

                        elif(scenejsonname=='watervessel0.9x.json'):
                            rotationself(wallmoveidx,box,center=tf.constant([2.7,0.0,2.7])*scale,vel=tempomega)

                        else:
                            rotationself(wallmoveidx,box,center=tf.constant([3.0,0.0,3.0])*scale,vel=tempomega)



                elif(scenejsonname=='wavetower.json'):
                    boardpartnum=wallinfo[1][0]

                    # pass                  
                    if((step%400)<=400/2):
                        #board
                        moverigid(wallmoveidx,boardpartnum,box,direct=-1.0)
                    else:
                        moverigid(wallmoveidx,boardpartnum,box)



                elif('wavetowerstatic' in scenejsonname):

                    scale = 1
                    if("1.3x" in scenejsonname):
                        scale=1.3

                    movetime=150
                    # movetime=999999

                    waittime=170
                    # waittime=500
                    # waittime=999999

                    #ie push at 680
                    # waittime=680 - movetime
                    waittime=770 - movetime
                    

                    pushperiod=700
                    wallmoveidx=wallinfo[1][0]+wallinfo[0][0]
                    towerpartnum=wallinfo[2][0]

                  
                    pushperiod=int(pushperiod*0.75)
                    

                    # no wait test
                    # pushperiod=int(pushperiod*0.6)

                    #MT DF MP
                    # pushperiod=int(pushperiod*0.5)


                    if(step<=movetime):
                        moverigid(wallmoveidx,towerpartnum,box,speed=-0.74*scale/movetime,axis=1)
                    elif(step<= movetime + waittime):
                        pass
                    else:
                        from movewall_strategy import movesin
                        wallmoveidx=wallinfo[0][0]
                        boardpartnum=wallinfo[1][0]
                        global boarddis
                        boarddis+=movesin(wallmoveidx,boardpartnum,box,(step-movetime-waittime+1),\
                        maxdis=4*scale,funcperiod=pushperiod)
                        print('dis---')
                        print(boarddis)



                elif(scenejsonname=='propeller.json'):
                    wallmoveidx=wallinfo[0][0]
                    proppartnum=wallinfo[1][0]
                    movetime=250
                    waittime=100


                    if(step<=movetime):
                        moverigid(wallmoveidx,proppartnum,box,speed=-1.8/movetime,axis=1)
                    elif(step<=movetime+waittime):
                        pass
                    else:
                        rotationself(wallmoveidx=wallmoveidx,\
                                    box=box,center=tf.constant([0.0,0.0,0.0]),
                                    vel=1.0/120.0,axis=0)


                elif(scenejsonname=='propellerlarge.json' or\
                     scenejsonname=='propellerlarge1.5x.json'):
                    wallmoveidx=wallinfo[0][0]
                    proppartnum=wallinfo[1][0]
                    movetime=250
                    waittime=100

                    boxscale=1
                    if(scenejsonname=='propellerlarge1.5x.json'):
                        boxscale=1.5


                    if(step<=movetime):
                        moverigid(wallmoveidx,proppartnum,box,speed=-1.7*boxscale/movetime,axis=1)
                    elif(step<=movetime+waittime):
                        pass
                    else:
                        rotationself(wallmoveidx=wallmoveidx,\
                                    box=box,center=tf.constant([0.0,-0.7,0.0])*boxscale,
                                    vel=-1.0/120.0,axis=0)
                
                elif(scenejsonname=='propeller2large.json'):
                    wallmoveidx=wallinfo[0][0]
                    proppartnum=wallinfo[1][0]
                    movetime=250
                    waittime=100


                    if(step<=movetime):
                        moverigid(wallmoveidx,proppartnum,box,speed=-0.6/movetime,axis=1)
                    elif(step<=movetime+waittime):
                        pass
                    else:
                        rotationself(wallmoveidx=wallmoveidx,\
                                    box=box,center=tf.constant([0.0,-0.2,0.0]),
                                    vel=-1.0/120.0,axis=1)

                elif(scenejsonname=='propeller2large2.json' or\
                     scenejsonname=='propeller2large21.5x.json' or \
                     scenejsonname=='propeller2+large2.json'or \
                     scenejsonname=='propeller2+large2fill.json'):
                        wallmoveidx=wallinfo[0][0]
                        proppartnum=wallinfo[1][0]

                        movetime=100
                        waittime=10

                        boxscale=1
                        
                        if(scenejsonname=='propeller2large21.5x.json'):
                            boxscale=1.5

                        if(scenejsonname=='propeller2+large2fill.json'):
                            proppartnum=wallinfo[1][0]+wallinfo[2][0]


                        # #test
                        rotscale=0.125
                
                        
                        if(step<=movetime):
                            moverigid(wallmoveidx,proppartnum,box,speed=-0.6*boxscale/movetime,axis=1)
                        elif(step<=movetime+waittime):
                            pass
                        else:
                            rotationself(wallmoveidx=wallmoveidx,\
                                        box=box,center=tf.constant([0.0,-0.2,0.0])*boxscale,
                                        vel=-1.0/120.0*rotscale,axis=1)

                elif(scenejsonname=='propellerbiglarge2.json'):
                        wallmoveidx=wallinfo[0][0]
                        proppartnum=wallinfo[1][0]
                        movetime=100
                        waittime=1

                        if(step<=movetime):
                            moverigid(wallmoveidx,proppartnum,box,speed=-1.55/movetime,axis=1)
                            
                        elif(step<=movetime+waittime):
                            pass
                        else:
                            # print(np.mean(box[wallmoveidx:,0]))
                            # print(np.mean(box[wallmoveidx:,1]))
                            # print(np.mean(box[wallmoveidx:,2]))
                            # assert(False)

                            rotationself(wallmoveidx=wallmoveidx,\
                                        box=box,center=tf.constant([0.04,-0.3,-0.07]),
                                        vel=-1.0/180.0,axis=1)


                elif(scenejsonname=='taylorvortex.json'or scenejsonname=='taylorvortex1.5scale.json'):
                    wallmoveidx=wallinfo[0][0]
                    proppartnum=wallinfo[1][0]
                    movetime=100
                    waittime=10

                    boxscale=1.0
                    if(scenejsonname=='taylorvortex1.5scale.json'):
                        boxscale=1.5
                        # assert(False)
                    


                    if(step<=movetime):
                        moverigid(wallmoveidx,proppartnum*4,box,speed=-0.6*boxscale/movetime,axis=1)
                    elif(step<=movetime+waittime):
                        pass
                    else:
                        
                        rotationself(wallmoveidx=wallmoveidx,\
                                    box=box,center=tf.constant([0.0,-0.2,0.0])*boxscale,
                                    vel=-1.0/120.0,axis=1,partnum=proppartnum)
                        rotationself(wallmoveidx=wallmoveidx+proppartnum,\
                                        box=box,center=tf.constant([-2.0,-0.2,2.43])*boxscale,
                                        vel=-1.0/120.0,axis=1,partnum=proppartnum)
                        rotationself(wallmoveidx=wallmoveidx+2*proppartnum,\
                                        box=box,center=tf.constant([0.0,-0.2,2.43])*boxscale,
                                        vel=1.0/120.0,axis=1,partnum=proppartnum)
                        rotationself(wallmoveidx=wallmoveidx+3*proppartnum,\
                                        box=box,center=tf.constant([-2.0,-0.2,0.0])*boxscale,
                                        vel=1.0/120.0,axis=1,partnum=proppartnum)






                elif(scenejsonname=='rotatingpanelstatic.json'
                or scenejsonname=='rotatingpanelstatic1.5x.json'
                or scenejsonname=='rotatingpanelstatic2x_thin.json'):

                    scenesize=1
                    if("1.5x" in scenejsonname):
                        scenesize=1.5
                    if("2x" in scenejsonname):
                        scenesize=2.0


                    wallmoveidx=wallinfo[0][0]
                    proppartnum=wallinfo[1][0]

                    statictime=0
                    # statictime=50
                    # statictime=120
                    
                    
                    movetime=50


                    waittime=10
                    # waittime=50
                    # waittime=99999


                    rotatetime=700
                    rotatetime=9999


                    if(step<statictime):
                        pass
                    elif(step<=statictime+movetime):

                        if("2x_thin" in scenejsonname):
                            moverigid(wallmoveidx,proppartnum*4,box,speed=-(0.3+0.2+0.5)*scenesize/movetime,axis=1)
                        else:
                            moverigid(wallmoveidx,proppartnum*4,box,speed=-(0.3+0.2)*scenesize/movetime,axis=1)

               

                    elif(step<=statictime+movetime+waittime):

                        pass

                    elif(step<=statictime+movetime+waittime+rotatetime):
                        #default        2.25deg/0.016s
                        rotscale=1.5

                        centx=np.mean(box[wallmoveidx:,0])
                        centy=np.mean(box[wallmoveidx:,1])
                        centz=np.mean(box[wallmoveidx:,2])

                        print(centx)
                        print(centy)
                        print(centz)

                        #rot0.75
                        # rotscale=0.75

                        # test
                        # rotscale=0.5
     

                        nowAngle=\
                        rotationself(wallmoveidx=wallmoveidx,\
                                    box=box,center=tf.constant([-1.0398,0.1935,1.1020])*scenesize,
                                    vel=-1.0*rotscale/120.0,axis=1,partnum=proppartnum,\
                                    # ratio=(step-movetime-waittime)/1000.0* 1000.0 /(1000.0-movetime-waittime)     #加速旋转acc
                                    
                                    
                                     )
                        angleeps=(np.mod(nowAngle, 2 * np.pi) - np.pi*3/2 )
                        if(angleeps<0 and abs(angleeps)<0.1):
                            print('[achieve 180 at frame] '+str(step))
                            print('[eps]\t'+str(angleeps))
                            rotationself(wallmoveidx=wallmoveidx,\
                                box=box,center=tf.constant([-1.0398,0.1935,1.1020])*scenesize,
                                vel=1 ,axis=1,partnum=proppartnum,\
                                )
                            
                    else:
                        moverigid(wallmoveidx,proppartnum*4,box,speed=(0.3+0.2)*scenesize/movetime,axis=1)







                # np.save(output_dir+"/rigid_"+str(step),box[wallmoveidx:])
                if(prmexportrig):
                    if(prmexportallrigidperframe):
                        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(box))
                    else:
                        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(box[wallmoveidx:]))

                    o3d.io.write_point_cloud(output_dir+"/rigid_{0:04d}".format(step)+ '.ply', pcd)

      


            if(prm_round):
                pos=np.round(pos,eps)
                vel=np.round(vel,eps)
            if(prm_mix):
                if((not prm_continuerun ) or\
                 (step>=prm_resumestep-1)):
                    pos, vel = model.call2(model2=model2,
                                        model3=model3,
                                        model4=model4,
                                        model5=model5, 
                                        inputs=inputs,
                                        step=step,
                                        num_steps=num_steps)
    
              
            else:
                # print('[pretype inputs]')
                # print(type(inputs[1]))#tensor

                                    
                # with tf.GradientTape() as tape:
                #     grads = tape.gradient(total_loss, model.trainable_variables)
                #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # # print(model.)


                if((not prm_continuerun ) or\
                 (step>=prm_resumestep-1)):
                    pos, vel = model(inputs)#numpy
                if(prm_round):
                    pos=np.round(pos,eps)
                    vel=np.round(vel,eps)


                #zxc 步长已经包含在model里了
        
        #zxc 或许不要更好
        # remove out of bounds particles

        if(prmsimplebc):        
            from simplebc import simplebc
            pos,vel=simplebc(pos,vel,min_x,min_y,min_z,max_x,max_y,max_z)
    
            vel=util.clearNan(vel)


        if step % 1 == 0:
            #prm_

            from tensorflow import logical_and,logical_not,where,expand_dims,tile,convert_to_tensor,cast,int32
            print(step, 'num particles', pos.shape[0])
            if(prm_mask):
                mask = pos[:, 1] > min_y
                mask = logical_and(mask,(pos[:, 2] > min_z))
                mask = logical_and(mask,(pos[:, 2] < max_z))
                mask = logical_and(mask,(pos[:, 0] > min_x))
                mask = logical_and(mask,(pos[:, 0] < max_x))


                model.mask=mask
                print('[MASK]')
                # mask = pos[:, 0]  < 8 #需要保留的粒子
                # if np.count_nonzero(mask) < pos.shape[0]:
                #     pos = pos[mask]
                #     vel = vel[mask]
            if(prm_moveOutPart):
                if(scenejsonname=="streammultiobjsHorizon.json"):
                    mask = pos[:, 0] > 21
                    if np.count_nonzero(mask):
                        print('moveOutPart..')
                        temp=pos.cpu().numpy()
                        temp[mask,0]=30
                        temp[mask,1]=30
                        temp[mask,2]=30
                        pos=temp

                elif(scenejsonname=='rotatingpanelstatic.json'):

                    mask = pos[:, 1] > 0.5
                    if np.count_nonzero(mask):
                        print('moveOutPart..')
                        temp=pos.cpu().numpy()
                        temp[mask,0]=30
                        temp[mask,1]=30
                        temp[mask,2]=30
                        pos=temp


                elif(scenejsonname=='mc_ball_2velx_0602_wake.json' 
                or scenejsonname=='mc_ball_2velx_0602_wake_wide.json'):
                    mask = pos[:,0] < -15       #right
                    mask = np.logical_or(mask,(pos[:, 0] > 6.4))
                    if(np.count_nonzero(mask)):
                        print('moveOutPart..')
                        temp=pos.cpu().numpy()
                        temp[mask,0]=30
                        temp[mask,1]=30
                        temp[mask,2]=30
                        pos=temp
               

                else:
                    print('no moveout part ways')
                    assert(False)


            if(prm_edit):
             

                mask = pos[:, 1]  > -2
                mask2= vel[:, 1]  > 0.01
          
              

                # print(mask.shape)
                # print(mask2.shape)
                # print(type(mask))

                # print(mask.dtype)
                # print(mask2.dtype)

                
                

                mask=logical_and(mask,mask2)
               
              
                print('--------mask part num-------------')
                print(np.sum(cast(mask,int32).cpu().numpy()))
                # mask= expand_dims(mask, axis=1)  
                # mask = tile(mask, [1, 3])
                print(mask.shape)#partnum 3

                velnp=vel.cpu().numpy()
                # posnp=pos.cpu().numpy()
                velnp[mask.cpu().numpy(),1] *= 0.6
                # vely=where(mask,np.zeros_like(vel[:,1]),vel[:,1])

                # vel=vel.cpu().numpy()
                # vel[:,1]=vely.cpu().numpy()

                vel=convert_to_tensor(velnp)
                # pos=convert_to_tensor(posnp)
                


                # print(x.shape)
                # mask=pos[:,1]
                # print(mask.shape)
                # vel[:,1]=x

                # vel[.cpu().numpy(),1]
                # assert(False)



    timeperframe=(time.time()-starttime)/num_steps
    print('[mtimes]\t'+str(model.mtimes))
    print('[cost]\t'+str(timeperframe)+'sec per frame\t')
    print('[infertime]\t'+str(model.infertime/num_steps) +'sec per frame\t')
    if(prm_mix):
        print('MIX')
    if(prm_continuerun):
        pass
        # data=np.load(output_dir + '.npz')
        # mat1=data['mat1']
        # mat2=data['mat2']
        # mat3=data['mat3']
        # mat4=data['mat4']
        # mat5=data['mat5']
        # mat6=data['mat6']
        # mat1_add=np.array(model.aenergy)
        # mat2_add=np.array(model.adelta_energy)
        # mat3_add=np.array(model.mtimes)
        # mat4_add=np.array(model.morder_pointwise if prm_pointwise else model.morder)
        # mat5_add=np.array(model.adelta_energy2)
        # mat6_add=np.array(model.acoff)

        # print(mat1_add.shape)

        # assert(False)

        # np.savez('temp.npz',\
        # mat1=np.concatenate((mat1, mat1_add), axis=0),\
        # mat2=np.concatenate((mat2, mat2_add), axis=0),\
        # mat3=np.concatenate((mat3, mat3_add), axis=0),\
        # mat4=np.concatenate((mat4, mat4_add), axis=0),\
        # mat5=np.concatenate((mat5, mat5_add), axis=0),\
        # mat6=np.concatenate((mat6, mat6_add), axis=0)
        
        # )

    else:

        print('[info saved]')
        np.savez(output_dir + '.npz',
            mat1=model.aenergy,\
            mat2=model.adelta_energy,\
            
            # 选择次数
            mat3=model.mtimes,
            mat4=(model.morder_pointwise if prm_pointwise else model.morder),
            mat5=model.adelta_energy2,
            
            # 混合系数
            mat6=model.acoff,
            mat7=model.acoff_area,
            mat8=model.acoff_other,
            mat9=model.aenergy_area,
            mat10=model.aenergy_other,
            mat11=model.apartnum_area,
            mat12=model.agammahat,
            mat13=model.agamma,
            mat14=model.aenergymax,
            mat15=model.aenergymin,
            mat16=model.aenergypre,
            mat17=model.agammahat2,
            mat18= model.afmax,
            mat19= model.afmin,
            mat20=model.afratio,
            mat21=model.afratioactual,
            mat22=model.aeratio,
            mat23=model.aeratioacual,
            mat24=  model.aemin,
            mat25=  model.aemax,
            amv=model.amv
         
            )
            
        #know


def run_sim_torch(trainscript_module, weights_path, scene, num_steps,
                  output_dir, options):
    import torch
    device = torch.device(options.device)

    # init the network
    model = trainscript_module.create_model()
    weights = torch.load(weights_path)



    weighttf=[]

    layername=['cvf','cvo','cv1','cv2','cv3','d0','d1','d2','d3']
    for i in layername:
        print('----------'+str(i)+'\t[layer]----------------')
        data=(np.load ('./weightnp/'+i+ '.npz'))
        print(data['weights'].shape)
        print(data['biases'].shape)
        weighttf.append(data)



    for k,v in weights.items():#know 视图view无法修改
        print('----------'+str(k)+'\t [torch layer]----------------')
        print(v.shape)
        # print(type(v))//tensor


    with torch.no_grad():  # 确保在不需要计算梯度的情况下设置权重  
      
        weights['conv0_fluid.kernel']=torch.from_numpy(weighttf[0]['weights'])
        weights['conv0_fluid.bias']=torch.from_numpy(weighttf[0]['biases'])


        weights['conv0_obstacle.kernel']=torch.from_numpy(weighttf[1]['weights'])
        weights['conv0_obstacle.bias']=torch.from_numpy(weighttf[1]['biases'])

        weights['conv1.kernel']=torch.from_numpy(weighttf[2]['weights'])
        weights['conv1.bias']=torch.from_numpy(weighttf[2]['biases'])

        weights['conv2.kernel']=torch.from_numpy(weighttf[3]['weights'])
        weights['conv2.bias']=torch.from_numpy(weighttf[3]['biases'])


        weights['conv3.kernel']=torch.from_numpy(weighttf[4]['weights'])
        weights['conv3.bias']=torch.from_numpy(weighttf[4]['biases'])

        weights['dense0_fluid.weight']=torch.from_numpy(weighttf[5]['weights'].T)
        weights['dense0_fluid.bias']=torch.from_numpy(weighttf[5]['biases'])

        weights['dense1.weight']=torch.from_numpy(weighttf[6]['weights'].T)
        weights['dense1.bias']=torch.from_numpy(weighttf[6]['biases'])

        
        weights['dense2.weight']=torch.from_numpy(weighttf[7]['weights'].T)
        weights['dense2.bias']=torch.from_numpy(weighttf[7]['biases'])

        weights['dense3.weight']=torch.from_numpy(weighttf[8]['weights'].T)
        weights['dense3.bias']=torch.from_numpy(weighttf[8]['biases'])

        print('[changed]')





    model.load_state_dict(weights)
    model.to(device)
    model.requires_grad_(False)

    if not os.path.exists('./cache/'+scenejsonname+'-f.npy'):#know
        print('[no scene ply cache]') 
        # prepare static particles
        walls = []
        for x in scene['walls']:
            points, normals = obj_surface_to_particles(x['path'])
            if 'invert_normals' in x and x['invert_normals']:
                normals = -normals
                print('[invert normal]')
            points += np.asarray([x['translation']], dtype=np.float32)
            walls.append((points, normals))
        box = np.concatenate([x[0] for x in walls], axis=0)
        box_normals = np.concatenate([x[1] for x in walls], axis=0)

        # export static particles
        write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

        # compute lowest point for removing out of bounds particles
        min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))



        # prepare fluids
        fluids = []
        for x in scene['fluids']:
            points = obj_volume_to_particles(x['path'])[0]
            points += np.asarray([x['translation']], dtype=np.float32)
            velocities = np.empty_like(points)
            velocities[:, 0] = x['velocity'][0]
            velocities[:, 1] = x['velocity'][1]
            velocities[:, 2] = x['velocity'][2]
            range_ = range(x['start'], x['stop'], x['step'])
            fluids.append(
                (points.astype(np.float32), velocities.astype(np.float32), range_))
            #zxc
        np.save('./cache/'+scenejsonname+"-f",fluids)
        np.save('./cache/'+scenejsonname+"-box",box)
        np.save('./cache/'+scenejsonname+"-boxn",box_normals)

     
    else:
        print('[use cache]')
        fluids=     np.load('./cache/'+scenejsonname+"-f.npy",allow_pickle=True)
        box=        np.load('./cache/'+scenejsonname+"-box.npy",allow_pickle=True)
        box_normals=np.load('./cache/'+scenejsonname+"-boxn.npy",allow_pickle=True)

        print(fluids.dtype)
        print(fluids.shape)

        # print(box.dtype)
        # print(box.shape)
        # assert(False)

    box = torch.from_numpy(box).to(device)  
    box_normals = torch.from_numpy(box_normals).to(device)



    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)

    for step in tqdm(range(num_steps)):
        # add from fluids to pos vel arrays
        for points, velocities, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel, options)
            else:
                write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                options)

            inputs = (torch.from_numpy(pos).to(device),
                      torch.from_numpy(vel).to(device), None, box, box_normals)
            pos, vel = model(inputs)
            pos = pos.cpu().numpy()
            vel = vel.cpu().numpy()

        # remove out of bounds particles
        if step % 10 == 0:
            print(step, 'num particles', pos.shape[0])
            mask = pos[:, 1] > min_y
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]


def main():
    parser = argparse.ArgumentParser(
        description=
        "Runs a fluid network on the given scene and saves the particle positions as npz sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript",
                        type=str,
                        help="The python training script.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=
        "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )
    parser.add_argument("--num_steps",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--scene",
                        type=str,
                        required=True,
                        help="A json file which describes the scene.")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory for the particle data.")
    parser.add_argument("--write-ply",
                        action='store_true',
                        help="Export particle data also as .ply sequence")
    parser.add_argument("--write-bgeo",
                        action='store_true',
                        help="Export particle data also as .bgeo sequence")
    parser.add_argument("--device",
                        type=str,
                        default='cuda',
                        help="The device to use. Applies only for torch.")

    parser.add_argument("--prm_mix",
                        type=bool,
                        default=False,
                        help="The device to use. Applies only for torch.")

    parser.add_argument("--prm_mixmodel",
                        type=str,
                        default="error",
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()
    print(args)
    global prm_mix,prm_mixmodel
    
    prm_mix=args.prm_mix
    prm_mixmodel=args.prm_mixmodel


    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript_module = importlib.import_module(module_name)
    global scenejsonname
    scenejsonname=args.scene
    with open(args.scene, 'r') as f:
        scene = json.load(f)

    if(not prm_continuerun):
        os.makedirs(args.output)

    if args.weights.endswith('.h5'):
        return run_sim_tf(trainscript_module, args.weights, scene,
                          args.num_steps, args.output, args)
    elif args.weights.endswith('.pt'):
        return run_sim_torch(trainscript_module, args.weights, scene,
                             args.num_steps, args.output, args)


if __name__ == '__main__':
    sys.exit(main())
