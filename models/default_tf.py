import tensorflow as tf
import time

from tuneCurve import *
from util import getnpz
from prms import *
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)


fe_inited=False


#如果同时需要运行taichi，要记得限制内存-------------
#太小有可能跑不了（哪怕没有taichi）
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], 
#    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*24)])




import os 
# 设置环境变量，只显示错误日志
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(type(os.environ["CUDA_VISIBLE_DEVICES"]))
# TEMPCUDA=os.environ["CUDA_VISIBLE_DEVICES"]
# os.environ["CUDA_VISIBLE_DEVICES"] = ""#使用CPU
# # 打印当前TensorFlow版本
# print("TensorFlow version:", tf.__version__)
# # 设置TensorFlow只使用CPU
# tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')





# from tensorflow.python.client import device_lib as _device_lib
# local_device_protos = _device_lib.list_local_devices()
# devices = [x.name for x in local_device_protos]
# for d in devices:
# 	print(d)
# assert(False)

#-------------------------------------------------


import open3d.ml.tf as ml3d
import numpy as np

import sys
sys.path.append('../scripts/') 
from train_network_tf import bvor,dt_frame
# np.set_printoptions(precision=100)

tempcnt=0
from energy import *
from prms import prm_maxenergy,prm_pointwise,prm_area,prm_sus,\
prm_linear,prm_customtune,prm_mlpexact,prm_mlpexact_2,\
prm_needratio,\
prm_energyratio,\
ratio0,\
prm_3dim,\
prm_exactarg,\
prm_slopeGravity,\
prm_0gravity,\
prmmixstable,\
prmstableratio,\
prmtune0,\
prmtuneend,\
prm_exportgap,\
prmfixcoff






if(prm_linear):
    if(prm_mlpexact):
        from mlp2_train_exact import predictincconv_exact
    elif(prm_mlpexact_2):
        from mlp2_train_exact___2 import predictincconv_exact,relumatrix

    else:
        from mlp2_train import predictincconv



# print('\n\n\n[def]\n\n\n')
# xx=tf.random.normal((1,1))
# print(xx)
# exit(0)

class MyParticleNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_size=[4, 4, 4],#整数，核的分辨率
                 radius_scale=1.5,
                 coordinate_mapping='ball_to_cube_volume_preserving',
                 interpolation='linear',
                 use_window=True,
                 particle_radius=0.025,
                 #prm
                 timestep=1 / 50,
                 #prm ie 0.02

                 gravity=(0, -9.81, 0)



                 #horizon
                #  gravity=(1.5, -9.6845, 0)

                 ):
                 #prm
        super().__init__(name=type(self).__name__)
        
        #zxc add
        self.mtimes=[0,0,0,0]
        self.aenergy=[]
        self.amv=[]
        self.adelta_energy=[]
        self.adelta_energy2=[]
        self.acoff=[]
        
        self.acoff_area=[]
        self.acoff_other=[]
        self.aenergy_area=[]
        self.aenergy_other=[]
        self.apartnum_area=[]
        self.agammahat=[]
        self.agamma=[]
        self.aenergymax=[]
        self.aenergymin=[]
        self.aenergypre=[]
        self.agammahat2=[]
        self.afmax=[]
        self.afmin=[]
        self.aemin=[]
        self.aemax=[]
      
        self.afratio=[]
        self.afratioactual=[]
        self.aeratio=[]
        self.aeratioacual=[]

        self.infertime=0

        self.prm_linear=prm_linear


        if(prm_slopeGravity):
            gravity=(1.5, -9.6845, 0)
        elif(prm_0gravity):
            gravity=(0.0,0.0,0.0)

         


        ID_ENERGY='mat1'
        self.aargmax=getnpz('cp__emax_b_csm_df300_1111mc_ball_2velx_0602')[ID_ENERGY][0:999]
        self.aargmin=getnpz('cp__emin_b_csm_df300_1111mc_ball_2velx_0602')[ID_ENERGY][0:999]
        print(self.aargmax.shape)#999,
        # assert(False)


        self.morder=[]
        self.morder_pointwise=[]
        self.correctmodel_pointwise=None
        self.modelnum=4

        self.gravity_vec=np.array([0,-9.81,0])

        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window

        
        self.particle_radius = particle_radius
        # if(bvor):
        #     print('de[bvor]')
        #     self.particle_radius=0.03


        self.filter_extent0 = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)

        if(prmFE):
            self.filter_extent = np.float32(prmFEval *
                                            self.particle_radius)
            
        else:
            self.filter_extent = self.filter_extent0

        fe_inited=True

            
        self.timestep = timestep
        if(bvor):
            self.timestep=dt_frame


        self.gravity = gravity

        self._all_convs = []

        def window_poly6(r_sqr):
            return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None,symmetric=False, **kwargs):



            # print("当前工作目录:", os.getcwd())

            if(prmdmcf):
                sys.path.append('../models/')
                from convolutionsdmcf import ContinuousConv
                conv_fn = ContinuousConv
                print('[DMCF]')
            else:
                conv_fn = ml3d.layers.ContinuousConv
                print('[CCONV]')



            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            # API
            if(prmdmcf):
                conv = conv_fn(name=name,
                            kernel_size=self.kernel_size,
                            activation=activation,
                            align_corners=True,
                            interpolation=self.interpolation,
                            coordinate_mapping=self.coordinate_mapping,
                            normalize=False,
                            window_function=window_fn,
                            radius_search_ignore_query_points=True,
                            symmetric=symmetric,
                            sym_axis=1,
                            **kwargs)     
            else:
                conv = conv_fn(name=name,
                                kernel_size=self.kernel_size,
                                activation=activation,
                                align_corners=True,
                                interpolation=self.interpolation,
                                coordinate_mapping=self.coordinate_mapping,
                                normalize=False,
                                window_function=window_fn,
                                radius_search_ignore_query_points=True,
                                **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                filters=self.layer_channels[0],
                                #32 channel
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = tf.keras.layers.Dense(name="dense0_fluid",
                                                  units=self.layer_channels[0],
                                                  activation=None)

        self.convs = []
        self.denses = []

        self.mask=-1
        #次级的卷积和全连接
        for i in range(1, len(self.layer_channels)):
            print('zxc layer channel')
            print(self.layer_channels[i])#64 64 3
            ch = self.layer_channels[i]
            dense = tf.keras.layers.Dense(units=ch,
                                          name="dense{0}".format(i),
                                          activation=None)

            # 多次调用 
            if(prmdmcf and i==len(self.layer_channels)-1):
                conv = Conv(name='conv{0}'.format(i), filters=ch, activation=None,symmetric=True)
                print('zxc build SYM kernel:\t'+str('conv{0}'.format(i)))
                # conv3
                # total trainable param num:68w
                # 不开启这个 ：69w
                
            else:
                conv = Conv(name='conv{0}'.format(i), filters=ch, activation=None)
            self.denses.append(dense)
            self.convs.append(conv)


        # conv = Conv(name='zxcsym', filters=ch, activation=None,symmetric=True)
        # self.convs.append(conv)
        # assert(False)
        


    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * tf.constant(self.gravity)
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # assert(fe_inited==True)
        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.filter_extent)
        filter_extent0 = tf.constant(self.filter_extent0)


        fluid_feats = [tf.ones_like(pos[:, 0:1]), vel]
        # print('zxc feat-------------')
        # print(fluid_feats[0].shape) 
        # print(fluid_feats[1].shape) 
        #N 1
        #N 3, N是变化的
        if not other_feats is None:
            fluid_feats.append(other_feats)
            #zxc

        fluid_feats = tf.concat(fluid_feats, axis=-1)


        #RS_1layer_(RSL)

        #zxc 这里才是正向执行，Init里只是搭建网络
        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent)
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent0)
        
        # 第一次concat
        
        # print(self.ans_conv0_fluid.shape)#partnum 32。卷积核数=32，依次作用于每一个粒子
        # print(self.ans_dense0_fluid.shape)
        # print(self.ans_conv0_obstacle.shape)

        # if(self.ans_conv0_fluid.shape[0]!=1):
        #     assert(False)
        
        # 第一层的输出
        feats = tf.concat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
                          axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = tf.keras.activations.relu(self.ans_convs[-1])
            # print('inp f')
            # print(inp_feats.shape)
            #partnum 96
            #partnum 64
            #partnum 64
            
            # https://www.open3d.org/docs/0.11.0/python_api/open3d.ml.tf.layers.ContinuousConv.html
            # 对应到call方法里的第四个参数
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)
        # assert(False)
        #zxc
        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            tf.ones_like(self.conv0_fluid.nns.neighbors_index,
                         dtype=tf.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        return self.pos_correction
        #zxc
    def call2(self,model2,model3,model4,model5,\
    inputs,step,num_steps, fixed_radius_search_hash_table=None):
        #zxc 前向过程
        
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.

        zxc
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs
        print(type(pos))#numpy
        print(type(vel))#numpy

        #第一次是，后面就不是了
        # if not isinstance(pos,np.ndarray):
        #     assert(False)


        # tf.convert_to_tensor(numpy_array)

        # assert(False)

        # _vel=vel.numpy()
        _vel=vel    #is numpy
        partnum=pos.shape[0]
        energy=getEnergy(vel=_vel,mask=self.mask,partnum=partnum)
        # if(not isinstance(self.mask, int)):
        #     print('legalpartnum\t'+str(np.sum(self.mask.cpu().numpy().astype(np.int))))
        tune=-1

        self.aenergy.append(energy)    
        # print(ensure_2d(mv.numpy()).shape) 
        # self.amv.append(ensure_2d(mv.numpy()))
        print('[energy]\t'+str(energy))

        #testterm toomuch
        # if(energy>30):
        #     print('too much energy')
        #     assert(False)

        


        #zxc 简单施加重力后的结果
        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        _vel2=vel2.numpy()

        #zxc 仅这一步用nn
        
        stm=time.time()

        pos_correction1 = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos_correction2=model2.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos_correction3=model3.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos_correction4=model4.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)

        self.infertime+=(time.time()-stm)

        global prmmixstable
        if(prmmixstable):
            assert(False)
            pos_correction5=model5.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)


        energynext1=np.sum(    getEnergynextBest(pos_correction=pos_correction1,dt_frame=dt_frame,_vel=_vel,gravity_vec=self.gravity_vec)  )/partnum
        energynext2=np.sum(    getEnergynextBest(pos_correction=pos_correction2,dt_frame=dt_frame,_vel=_vel,gravity_vec=self.gravity_vec)  )/partnum
        energynext3=np.sum(    getEnergynextBest(pos_correction=pos_correction3,dt_frame=dt_frame,_vel=_vel,gravity_vec=self.gravity_vec)  )/partnum
        energynext4=np.sum(    getEnergynextBest(pos_correction=pos_correction4,dt_frame=dt_frame,_vel=_vel,gravity_vec=self.gravity_vec)  )/partnum


        energynexts=[energynext1,energynext2,energynext3,energynext4]


        alpha=0.5
        global tempcnt
        tempcnt+=1
        ratio=step/num_steps
        
        alpha=float(ratio)
        alpha=float(ratio)**3
        

        global prm_maxenergy



        if(prm_sus):
            sustime=650


            print('[SUS]\t'+str(sustime))
            if(step<=sustime):
                print('[In sus]')
                prmmixstable=0
                prm_maxenergy=0
                self.prm_linear=0

                
                
            else:
                print('[sus end]')
                prmmixstable=0
                prm_maxenergy=0
                self.prm_linear=1

               
                
            # if(step<=500 or (step>=900)):
            #     prm_maxenergy=1
            # else:
            #     prm_maxenergy=0


        
        if(step%50==0):
            print('[alpha]\t'+str(alpha))

            #一次会输出一个场景中所有位置的矫正
            # temp=pos_correction1.cpu().numpy()
            # print(temp.shape)
        # pos_correction=(1-alpha)*(pos_correction1)+alpha*pos_correction2

        pos_corrections=[pos_correction1,pos_correction2,pos_correction3,pos_correction4]





        _pos_correction1=pos_corrections[1-1].cpu().numpy()
        _pos_correction2=pos_corrections[2-1].cpu().numpy()
        _pos_correction3=pos_corrections[3-1].cpu().numpy()
        _pos_correction4=pos_corrections[4-1].cpu().numpy()


        dv1=_pos_correction1/self.timestep
        dv2=_pos_correction2/self.timestep
        dv3=_pos_correction3/self.timestep
        dv4=_pos_correction4/self.timestep

        
        delta_energy_mat1=0
        delta_energy_mat2=0
        delta_energy_mat3=0
        delta_energy_mat4=0
        delta_energy_mat1=np.sum(getDeltaEnergy(v=_vel2,dv=dv1),axis=1)#know 沿着某个维度求和
        delta_energy_mat2=np.sum(getDeltaEnergy(v=_vel2,dv=dv2),axis=1)
        delta_energy_mat3=np.sum(getDeltaEnergy(v=_vel2,dv=dv3),axis=1)
        delta_energy_mat4=np.sum(getDeltaEnergy(v=_vel2,dv=dv4),axis=1)

        delta_energy_mat=[]
        delta_energy_mat.append(delta_energy_mat1)
        delta_energy_mat.append(delta_energy_mat2)
        delta_energy_mat.append(delta_energy_mat3)
        delta_energy_mat.append(delta_energy_mat4)

      
        
        if(prm_pointwise):



            self.correctmodel_pointwise=np.zeros_like(delta_energy_mat[1])
            # print('[270]')
            # print(delta_energy_mat[1].shape)#partnum
            # exit(0)
            bool_corrections=[]
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))
            bool_corrections.append(np.zeros_like(delta_energy_mat[1]))

            # print('[277]')
            # print(bool_corrections[0].shape)#partnum




            #找出每个点修正的量来自于哪个模型
            if(prm_maxenergy):

                temp=tf.maximum(delta_energy_mat[1-1],delta_energy_mat[2-1])
                temp=tf.maximum(temp,delta_energy_mat[3-1])
                temp=tf.maximum(temp,delta_energy_mat[4-1])

            else:
                temp=tf.minimum(delta_energy_mat[1-1],delta_energy_mat[2-1])
                temp=tf.minimum(temp,delta_energy_mat[3-1])
                temp=tf.minimum(temp,delta_energy_mat[4-1])
                # assert(False)


                #         (temp-delta_energy_mat[modelidx])
                # (逐点速度-速度矩阵1)。
                # 从能量反推是来自哪个模型

            bool_corrections[0]=tf.less(tf.abs(temp-delta_energy_mat[1-1]),1e-5)
            bool_corrections[1]=tf.less(tf.abs(temp-delta_energy_mat[2-1]),1e-5)
            bool_corrections[2]=tf.less(tf.abs(temp-delta_energy_mat[3-1]),1e-5)
            bool_corrections[3]=tf.less(tf.abs(temp-delta_energy_mat[4-1]),1e-5)

            # tf.cast(tf.less(tf.abs(temp-delta_energy_mat[1-1]),1e-5),tf.int32)

          

            print(self.correctmodel_pointwise[:100])
         
            print(self.correctmodel_pointwise.dtype)
            print(type(self.correctmodel_pointwise))

            print('[mid pw]')
            print(self.correctmodel_pointwise)
            # assert(False)
            
            # for x in range(0,delta_energy_mat[1-1].shape[0]):
            #         for modelidx in range(0,self.modelnum):
            #             if(abs(temp[x]-delta_energy_mat[modelidx][x])<1e-6):
            #                 bool_corrections[modelidx][x]=1
            #                 self.correctmodel_pointwise[x]=modelidx
            #                 break
            # self.correctmodel_pointwise=self.correctmodel_pointwise.astype(np.int8)
            # print(bool_corrections[0].shape)

            for modelidx in range(0,self.modelnum):
                # bool_corrections[modelidx]=bool_corrections[modelidx].reshape(-1,1)#partnum 1 know
                bool_corrections[modelidx]=tf.reshape(bool_corrections[modelidx],[-1,1])

                bool_corrections[modelidx]=np.tile(bool_corrections[modelidx], (1, 3)) #partnum 3 know

            # print('[tile]')
            # print(bool_corrections[0].shape)



                

        # print('[300]')
        # print(((dv1**2)+2*_vel2*dv1).shape)#partnum 3

        delta_energy1=0
        delta_energy2=0
        delta_energy3=0
        delta_energy4=0


        delta_energy1=np.sum(delta_energy_mat1)
        delta_energy2=np.sum(delta_energy_mat2)
        delta_energy3=np.sum(delta_energy_mat3)
        delta_energy4=np.sum(delta_energy_mat4)
      

        if(self.prm_linear and (not prm_area)):
            # tune=0.5
            # if(step>=300):

            
            #curve
            tune0=prmtune0
            tuneend=prmtuneend
            
            tune=x1(tune0=tune0,tuneend=tuneend,step=step,num_steps=num_steps)

            # tune=oscillating(t=step,y0=tune0,y1=tuneend,t0=num_steps,n=6)


                
            if(prm_customtune):
                tune=np.load('tunecurve.npy')[step]
            #line
            # tune=((num_steps-step)**1)*tune0/num_steps**1
            # if(step>1000):
                # tune=tuneend

            print('[now tune]\t'+str(tune))

            # if(step<=100):
            #     tune=0.2
            # elif(step<=400):
            #     tune=0.4   
            # elif(step<=700):
            #     tune=0.6
            # else:
            #     tune=1

            


            #exact 
            if(prm_mlpexact):
                pos_correction1perpart=tf.reduce_mean(pos_correction1,axis=0)
                pos_correction2perpart=tf.reduce_mean(pos_correction2,axis=0)
                pos_correction3perpart=tf.reduce_mean(pos_correction3,axis=0)
                pos_correction4perpart=tf.reduce_mean(pos_correction4,axis=0)

               

                coff=predictincconv_exact(
                    d1=pos_correction1perpart,
                    d2=pos_correction2perpart,
                    d3=pos_correction3perpart,
                    d4=pos_correction4perpart,
                    v_t=tf.reduce_mean(_vel,axis=0),
                    gt_d2=self.gravity_vec*dt_frame/2,
                    partnum=partnum,
                    tune=tune
                    
                )
            elif(prm_mlpexact_2):

                #把所有粒子的求和
                if(prm_3dim):
                    pos_correction1perpart=tf.reduce_mean(pos_correction1,axis=0)
                    pos_correction2perpart=tf.reduce_mean(pos_correction2,axis=0)
                    pos_correction3perpart=tf.reduce_mean(pos_correction3,axis=0)
                    pos_correction4perpart=tf.reduce_mean(pos_correction4,axis=0)
                    xraw=np.array([
                            np.array(pos_correction1perpart/dt_frame),\
                            np.array(pos_correction2perpart/dt_frame),\
                            np.array(pos_correction3perpart/dt_frame),\
                            np.array(pos_correction4perpart/dt_frame),
                            np.array([np.mean(_vel[:,0]),\
                                    np.mean(_vel[:,1]),\
                                    np.mean(_vel[:,2])]),
                            np.array(self.gravity_vec*dt_frame/2)
                    ]
                            )
                #平均速度大小
                else:
                    pos_correction1_normav=tf.reduce_mean(tf.sqrt(tf.reduce_sum(pos_correction1**2,axis=1)))
                    pos_correction2_normav=tf.reduce_mean(tf.sqrt(tf.reduce_sum(pos_correction2**2,axis=1)))
                    pos_correction3_normav=tf.reduce_mean(tf.sqrt(tf.reduce_sum(pos_correction3**2,axis=1)))
                    pos_correction4_normav=tf.reduce_mean(tf.sqrt(tf.reduce_sum(pos_correction4**2,axis=1)))
                    velnorm=np.mean(np.sqrt(np.sum(_vel**2,axis=1)))
                    xraw=np.array([
                            np.array([pos_correction1_normav/dt_frame]),\
                            np.array([pos_correction2_normav/dt_frame]),\
                            np.array([pos_correction3_normav/dt_frame]),\
                            np.array([pos_correction4_normav/dt_frame]),
                            np.array([velnorm]),
                            np.array([-9.81*dt_frame/2])
                    ]
                            )
                # print(pos_correction1perpart.shape)#3,

                xraw=xraw.T
                print(xraw.shape)#samplenum 6
                # assert(False)

                #COPY
                fmin=[]
                fmax=[]
                for i in range(xraw.shape[0]):
                    from util import relu
                    a1=relu ( xraw[i,0])+relu( xraw[i,1])+relu( xraw[i,2])+relu( xraw[i,3])      +xraw[i,4]+xraw[i,5]
                    a2=-relu(-xraw[i,0])-relu(-xraw[i,1])-relu(-xraw[i,2])-relu(-xraw[i,3])      +xraw[i,4]+xraw[i,5]
                    fmax.append(a1)
                    fmin.append(a2)
                fmax=np.array(fmax)[:,np.newaxis].T
                fmin=np.array(fmin)[:,np.newaxis].T
                # print(fmax.shape)
                # print(fmin.shape)#1 3
                # assert(False)


                fmin2=fmin**2
                fmax2=fmax**2


                mine=np.minimum(fmin**2,fmax**2)#1 3
                maxe=np.maximum(fmin**2,fmax**2)#1 3
                for i in range(fmax.shape[1]):
                    if(fmax[0,i]*fmin[0,i]<0):
                        print('f range reverse')
                        #下一帧速度的范围可以翻转
                        mine[0,i]=0


             
                # print(maxe.shape)
                


                print('[f min,max]'+str([fmin,fmax]))



                
                # needf0=np.sqrt((self.aargmax[step]-self.aargmin[step])*0.5+self.aargmin[step])
                
                if(prm_needratio):
                    
                    if(prm_energyratio):    #指定网络需要达到的能量比例
                        neede=ratio0*(maxe-mine)+mine
                        needf=np.sqrt(neede)#为了达到这个能量，速度应该是多少
                        fratio1=( needf-fmin)/(fmax-fmin)

                        fratio2=(-needf-fmin)/(fmax-fmin)

                       

                        fratio=np.zeros_like(fratio1)
                        # print(fratio.shape)#1,3
                        # assert(False)
                        for i in range(0,fratio1.shape[1]):
                             # 哪个速度变化更合理，就用哪个
                            if(abs(fratio1[0,i]-0.5)<abs(fratio2[0,i]-0.5)):
                                fratio[0,i]=fratio1[0,i]
                            else:
                                fratio[0,i]=fratio2[0,i]

                        # print(fratio.shape)#1 3
                        # fratio=fratio.T
                   
                        # assert(False)
                    else:
                        fratio=ratio0
                        needf=(fmax-fmin)*fratio+fmin
                    
                else:
                    needf=needf0    #直接指定网络需要达到的F
                


                  
                    for i in range(fmax.shape[1]):
                        needratio[i]=(needf-fmin[:,i])/(fmax[:,i]-fmin[:,i])
                        print('needratio\t'+str(needratio[i]))
                        if(needratio[i]>1.5 ):
                            print('in capability')
                            needratio[i]=1.5
                        if(needratio[i]<0 ):
                            print('in capability')
                            needratio[i]=0                
                  
                            

                

          
                       
           

                tune=fratio

                assert(False)

                k=1/np.amax(abs(xraw),axis=1)
                k=k[...,np.newaxis]
                xnorm=k*xraw
                # fratio=np.array([fratio,fratio,fratio])
                # print(fratio.shape)
                # print(xnorm.shape)
             
                # assert(False)

                coff=predictincconv_exact(
                    d1=xnorm[:,0],
                    d2=xnorm[:,1],
                    d3=xnorm[:,2],
                    d4=xnorm[:,3],
                    v_t=xnorm[:,4],
                    gt_d2=xnorm[:,5],
                    partnum=partnum,
                    tune=fratio
                    
                )
                assert(False)

            

            else:
                coff=predictincconv(delta_energy1,delta_energy2,delta_energy3,delta_energy4,partnum,tune)
                print(tf.reduce_mean(pos_correction1,axis=0).shape)#(3,)


                if(prmfixcoff):

                    coff=np.ones_like(coff)
                    if(step>=500 and step<=800):
                        coff*=(0.187474-0.1)*(step-500)/(500-800)+0.187474
                    elif(step>=800):
                        coff*=0.1
                    else:
                        assert(False)

                # assert(False)

            print('coff:')

            print(coff.shape)#3,9
            print(coff)

            # print('dE:')
            # temp=[delta_energy1,delta_energy2,delta_energy3,delta_energy4]
            # # print(temp.shape)
            # print(temp)

            # print(coff*delt)
        #prm_
        delta_energys=np.array([delta_energy1,delta_energy2,delta_energy3,delta_energy4])

        #case-B
        # delta_energys=np.array([delta_energy1,delta_energy2,             ,delta_energy4])

        idxmin=np.argmin(delta_energys)
        idxmax=np.argmax(delta_energys)

        if(prm_exactarg):
            idxmin=np.argmin(energynexts)
            idxmax=np.argmax(energynexts)



        #testterm 1m
        # idxmax=0
        # idxmin=0
        

        # print('[delta E]\t'+str(delta_energy[1-1]))

        if(prm_maxenergy and not self.prm_linear):
            pos_correction= pos_corrections[idxmax]   
            if(prmmixstable):
                print('STABLE\t'+str(prmstableratio))
                pos_correction=prmstableratio    *pos_correction5 +\
                               (1-prmstableratio)*pos_correction



            print('[choose]\t'+str(idxmax)) 
            self.morder.append(idxmax)
            self.mtimes[idxmax]+=1

        elif(not prm_maxenergy and not self.prm_linear):
            pos_correction= pos_corrections[idxmin]
            # if(step>=160):
            # if(step>=150):
            #     pos_correction= pos_corrections[2]
            #     print('artifical 2')

            if(prmmixstable):
                print('STABLE\t'+str(prmstableratio))
                pos_correction=prmstableratio    *pos_correction5 +\
                               (1-prmstableratio)*pos_correction
                # assert(False)
            
            print('[choose]\t'+str(idxmin)) 
            self.morder.append(idxmin)
            self.mtimes[idxmin]+=1
                     
        if(self.prm_linear and not(prm_area)):
            if(not prm_mlpexact):
                # coff=np.random.dirichlet(np.array([1.0, 1.0, 1.0, 1.0]))  
                # print(coff.shape)
                # assert(False)
                pos_correction=pos_corrections[0]*coff[0,0]+\
                pos_corrections[1]*coff[0,1]+\
                pos_corrections[2]*coff[0,2]+\
                pos_corrections[3]*coff[0,3]

           

            #EXACt
            else:
                print(pos_corrections[0].shape)#partnum 3
                print(coff[:,0].shape)#(1,)
                coff=coff[:,np.newaxis]#3 1 9
                print(coff.shape)
                # assert(False)
                pos_correction=pos_corrections[0]*coff[:,:,0].T+\
                            pos_corrections[1]*coff[:,:,1].T+\
                            pos_corrections[2]*coff[:,:,2].T+\
                            pos_corrections[3]*coff[:,:,3].T


        if(prm_area):


            print('----------step\t'+str(step))
            
            # print(pos.shape)#partnum 3
            # print(area_mask.shape)#partnum
            # exit(0)
            # 遍历每个点，检查其x坐标是否小于0.5  


            #prm_
            print('filtering...')
            area_mask = pos[:, 2] < 0 

            # area2_mask = pos[:,0] <-12
            area2_mask = pos[:,0] <-100

            if not isinstance(area_mask,np.ndarray):
                area_mask=area_mask.cpu().numpy() 
                area2_mask=area2_mask.cpu().numpy()

            # temp= pos[:, 0] < -2
            # if not isinstance(temp,     np.ndarray):
            #     temp=temp.cpu().numpy() 
            # area_mask *= temp

            print(area_mask.shape)#dynamic
            

            
                # one=tf.ones_like(area_mask)
            area_mask_bool=area_mask
            area_mask=area_mask.astype(np.float32)
            area2_mask=area2_mask.astype(np.float32)

            # print(area_mask_bool.dtype)

            # print(area_mask.dtype)
            # print(area_mask_bool)
            # print(np.logical_not(area_mask_bool))

            # assert(False)
            
            partnum_area=np.sum(area_mask)
            print('[partnum_area]\t'+str(partnum_area))
            # print()

            one=np.ones_like(area_mask)
            # print(area_mask)
            
            print('[done filter]')
            # print(temp)#partnum
            # print(area_mask.shape)#partnum 1
            # print(type(area_mask))#ndarray
            # print(area_mask.dtype)#bool
            # print(type(delta_energy_mat1))#ndarray

      
            area2_mask=np.array([area2_mask]).T
            area_mask=np.array([area_mask]).T
            one=np.array([one]).T
            delta_energy_mat1=np.array([delta_energy_mat1]).T
            delta_energy_mat2=np.array([delta_energy_mat2]).T
            delta_energy_mat3=np.array([delta_energy_mat3]).T
            delta_energy_mat4=np.array([delta_energy_mat4]).T



            


            print(delta_energy_mat1.shape)#partnum,1

        


            print(area_mask.dtype)#bool



            print('[z0]')
            # area_mask=tf.convert_to_tensor(area_mask)
            # delta_energy_mat1=tf.convert_to_tensor(delta_energy_mat1)
            
  
            delta_energy1_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat1).cpu().numpy())
            delta_energy2_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat2).cpu().numpy())
            delta_energy3_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat3).cpu().numpy())
            delta_energy4_area=np.sum(tf.math.multiply(area_mask,delta_energy_mat4).cpu().numpy())


            delta_energy1_area2=np.sum(tf.math.multiply(area2_mask,delta_energy_mat1).cpu().numpy())
            delta_energy2_area2=np.sum(tf.math.multiply(area2_mask,delta_energy_mat2).cpu().numpy())
            delta_energy3_area2=np.sum(tf.math.multiply(area2_mask,delta_energy_mat3).cpu().numpy())
            delta_energy4_area2=np.sum(tf.math.multiply(area2_mask,delta_energy_mat4).cpu().numpy())

        
            delta_energy1_other=delta_energy1-delta_energy1_area
            print('[area delta energs]')
            print(delta_energy1_area)
            print(delta_energy1_other)
            print(np.sum(delta_energy_mat1))

            delta_energy2_other =delta_energy2 - delta_energy2_area
            delta_energy3_other =delta_energy3 - delta_energy3_area
            delta_energy4_other =delta_energy4 - delta_energy4_area


            # if(step>0):#否则本来就是numpy
            #     if(isinstance(area_mask, np.ndarray)):
            #         pass
            #     else:

            #         area_mask=area_mask.cpu().numpy()

            # delta_energy_mat1=delta_energy_mat1.cpu().numpy()
            print('[z1]')

            tune0area=0.2
            tuneareaend=0.2

            

            tuneother0=0.5
            tuneotherend=0.5
            tune_area =x1(tune0=tune0area,tuneend=tuneareaend,step=step,num_steps=num_steps)

            # if(step>=250):
            #     tune_area=0.7
            tune_other=x1(tune0=tuneother0,tuneend=tuneotherend,step=step,num_steps=num_steps)
            #prm
            if(prm_maxenergy):
                idx_area=np.argmin([delta_energy1_area,\
                                    delta_energy2_area,\
                                    delta_energy3_area,\
                                    delta_energy4_area])
            else:
                idx_area=np.argmax([delta_energy1_area,\
                                    delta_energy2_area,\
                                    delta_energy3_area,\
                                    delta_energy4_area])
            if(self.prm_linear):

     

                coff_area= predictincconv(delta_energy1_area/partnum_area,\
                                          delta_energy2_area/partnum_area,\
                                          delta_energy3_area/partnum_area,\
                                          delta_energy4_area/partnum_area,\
                                          partnum_area,\
                                          tune_area)
                partnum_other=partnum-partnum_area
        
                coff_other=predictincconv(delta_energy1_other/partnum_other,\
                                          delta_energy2_other/partnum_other,\
                                          delta_energy3_other/partnum_other,\
                                          delta_energy4_other/partnum_other,\
                                          partnum_other,\
                                          tune_other)
                print('[coff]')                        
                print(coff_area)
                print(coff_other)

            if(idx_area==1-1):
                pos_correction_area=pos_correction1
            elif(idx_area==2-1):
                pos_correction_area=pos_correction2
            elif(idx_area==3-1):
                pos_correction_area=pos_correction3
            elif(idx_area==4-1):
                pos_correction_area=pos_correction4
            print('[area model]\t'+str(idx_area))
            print('[total model]\t'+str(idxmin))
            # print('[z2]')
            # print(area_mask.shape)
            area_mask_tile=np.tile(area_mask,[1,3])
            area2_mask_tile=np.tile(area2_mask,[1,3])

            one_tile=np.tile(one,[1,3])
            # print(area_mask.shape)
            # print(pos_correction.shape)s
            print(type(pos_correction))

            if(not isinstance(pos_correction,np.ndarray)):
                area_mask_tile=tf.convert_to_tensor(area_mask_tile)
                one_tile=tf.convert_to_tensor(one_tile)


            if(not self.prm_linear):
                pos_correction=\
                (one_tile-area_mask_tile)*(
                    pos_correction
                )+\
                area_mask_tile* pos_correction_area

            

            else:
                print(area2_mask_tile.shape)#partnum 3
                pos_correction=\
                (one_tile-area_mask_tile)*(one_tile-area2_mask_tile)*(
               
                coff_other[0,0]*pos_corrections[0]+\
                coff_other[0,1]*pos_corrections[1]+\
                coff_other[0,2]*pos_corrections[2]+\
                coff_other[0,3]*pos_corrections[3]


                )+\
                (area2_mask_tile)*pos_corrections[idxmin]+\
                area_mask_tile*(one_tile-area2_mask_tile)*(

                coff_area[0,0]*pos_corrections[0]+\
                coff_area[0,1]*pos_corrections[1]+\
                coff_area[0,2]*pos_corrections[2]+\
                coff_area[0,3]*pos_corrections[3]
                )
                # print(pos_corrections[1].shape)#partnum 3
                # print(area_mask_tile.shape)#partnum 3
                # print(one_tile.shape)#partnum 3

                # assert(False)

        if(prm_pointwise):
            # print(bool_corrections[0].shape)#partnum 3
            # print(type(bool_corrections[0]))#numpy
            # print(pos_correction1.shape)#partnum 3
            # print(type(pos_correction1))

            pos_correction=bool_corrections[1-1]*pos_correction1+\
                           bool_corrections[2-1]*pos_correction2+\
                           bool_corrections[3-1]*pos_correction3+\
                           bool_corrections[4-1]*pos_correction4
            pos_correction*=0.2

            # self.morder_pointwise.append(self.correctmodel_pointwise)

            print('model 1 correct num'+str(np.sum(bool_corrections[1-1][:,0])))
            print('model 2 correct num'+str(np.sum(bool_corrections[2-1][:,0])))
            print('model 3 correct num'+str(np.sum(bool_corrections[3-1][:,0])))
            print('model 4 correct num'+str(np.sum(bool_corrections[4-1][:,0])))

            # pos_correction=correct_pointwise


        #record
        print('[record]')
        if(prm_pointwise):
            self.morder_pointwise.append(
            [
            np.sum(bool_corrections[1-1][:,0]),
            np.sum(bool_corrections[2-1][:,0]),
            np.sum(bool_corrections[3-1][:,0]),
            np.sum(bool_corrections[4-1][:,0]),
            ]

            )
        elif(prm_area and (not self.prm_linear)):
            
            # exit(0)
            temp=\
            (one-area_mask)*(
                np.array([delta_energy_mat[idxmin]]).T
            )+\
            area_mask* np.array([delta_energy_mat[idx_area]]).T
            temp=np.sum(temp)/partnum
            self.adelta_energy.append(temp)
            print('[rec2]')
        elif(prm_maxenergy):
            self.adelta_energy.append(delta_energys[idxmax]/partnum)
        else:
            self.adelta_energy.append(delta_energys[idxmin]/partnum)
        if(self.prm_linear):
            if(prm_area):
                self.acoff_area. append(coff_area)
                self.acoff_other.append(coff_other)

                print('[tune area]\t'+(str(tune_area)))
                print('[tune other]\t'+(str(tune_other)))
                print(partnum_area)
                print(np.sum(area_mask))
                print(partnum)
                energy_area= getEnergy(vel=_vel,mask=area_mask_bool,                partnum=partnum)
                energy_other=getEnergy(vel=_vel,mask=np.logical_not(area_mask_bool),partnum=partnum)
                print('[area energy]')
                print(energy_area)
                print(energy_other)
                print(energy)

                # print(energy.shape)
                # print(area_mask.shape)
                # assert(False)
                # print(area_mask.dtype)
                # print((one-area_mask).dtype)
                # assert(False)
                self.aenergy_area.append (energy_area)
                self.aenergy_other.append(energy_other)
                self.apartnum_area.append(partnum_area)
            else:
                self.acoff.append(coff)
        
            

        gravity_mat=np.tile(self.gravity_vec,(partnum,1))
        # self.adelta_energy2.append(np.sum(getDeltaEnergy2(
        #             dt=self.timestep,
        #             dx=pos_correction,
        #             v=vel,
        #             g=gravity_mat)
        #         )/partnum)

        
     
        # print(type(self.gravity))
        # print(self.gravity.shape)
        # print(self.gravity.dtype)
        



   
        energynext= np.sum(    getEnergynextBest(pos_correction=pos_correction, dt_frame=dt_frame,_vel=_vel,gravity_vec=self.gravity_vec)  )/partnum

        if(prm_mlpexact_2):
            fnext=pos_correction/dt_frame + _vel + self.gravity_vec*dt_frame/2
            fnext=np.mean(fnext,axis=0)
            # print(k.shape)
            # print(fnext.shape)
            eesqr=k*fnext[:,np.newaxis]
            ffmax=np.sum(relumatrix(xnorm[...,0:4]               ),               axis=1)+xnorm[...,4]+xnorm[...,5]
            ffmin=np.sum(relumatrix(xnorm[...,0:4],positive=False),               axis=1)+xnorm[...,4]+xnorm[...,5]
            #验证EE是不是真的达到了对网络所提出的要求(检查网络本身的可靠性，和原始数据无关)
            # print(ffmax.shape)
            # print(eesqr.shape)

            print('------coff---------')
            print(coff[0,:4])
            print('----------input------')
            print(xnorm[0,:])
            print('-----------FF max,min------------------')
            print(ffmax)
            print(ffmin)
            # print(eesqr.shape)#3,1
            # print(ffmax.shape)#3

            # assert(False)
            print(ffmax.shape)#3
            print(fmax.shape)#1,3


            # print('-----------verify F ratio--------------------')和FF完全一致
            # print((np.sum(coff[0,:4]*xraw[0,:4])+xraw[0,4]+xraw[0,5]-(fmin.T)[:,0] [0])/((fmax.T)[:,0][0]-(fmin.T)[:,0][0]))
            # print((np.sum(coff[1,:4]*xraw[1,:4])+xraw[1,4]+xraw[1,5]-(fmin.T)[:,0] [1])/((fmax.T)[:,0][1]-(fmin.T)[:,0][1]))
            # print((np.sum(coff[2,:4]*xraw[2,:4])+xraw[2,4]+xraw[2,5]-(fmin.T)[:,0] [2])/((fmax.T)[:,0][2]-(fmin.T)[:,0][2]))

            print('-----------verify FF ratio--------------------')
            print((np.sum(coff[0,:4]*xnorm[0,:4])+xnorm[0,4]+xnorm[0,5]-ffmin[0])/(ffmax[0]-ffmin[0]))
            if(prm_3dim):
                print((np.sum(coff[1,:4]*xnorm[1,:4])+xnorm[1,4]+xnorm[1,5]-ffmin[1])/(ffmax[1]-ffmin[1]))
                print((np.sum(coff[2,:4]*xnorm[2,:4])+xnorm[2,4]+xnorm[2,5]-ffmin[2])/(ffmax[2]-ffmin[2]))
            print('[need F ratio]\t'+str(fratio))
            print('[need F]\t'+str(needf))
            print('f\t'+str(fnext))
            fnorm=np.sqrt(np.sum(fnext**2))
            print('f norm\t'+str(fnorm))
            print('f need\t'+str(needf))

            print('fmax\t'+str(fmax))
            print('fmin\t'+str(fmin))
            self.afmax.append(fmax)
            self.afmin.append(fmin)

            if(prm_3dim):
                fratioactual=(fnext-fmin)/(fmax-fmin)
            else:
                fratioactual=(fnorm-fmin)/(fmax-fmin)

            print('F ratio\t'+str(fratioactual))
            self.afratioactual.append(fratioactual)

            print('[need E ratio]\t'+str(ratio0))
            self.aeratio.append(ratio0)

            print('[need E]\t'+str(neede))
            # print('[verify tune]:'+str((eesqr-ffmin[:,np.newaxis])/(ffmax[:,np.newaxis]-ffmin[:,np.newaxis])))
            
            self.aemin.append(mine)
            self.aemax.append(maxe)
        
            gammahat2=(energynext-mine)/(maxe-mine)
            print('[gamma hat2]\t'+str(gammahat2))
            self.agammahat2.append(gammahat2)

        energymin=np.min(energynexts)
        energymax=np.max(energynexts)
        gammahat= (energynext-energymin)/(energymax-energymin)
        print('next single E max\t'+str(energymax))
        print('next single E min\t'+str(energymin))
        print('final choose s E\t'+str(energynext))

        
   


     
        # f2min=np.amax(fmin**2,fmax**2,axis=1)
        # f2max=np.amin(fmin**2,fmax**2,axis=1)

        # print('F2 ratio\t'+str((fnext**2-f2min)/(f2max-f2min)))



        if(self.prm_linear):
            print('[gamma]\t'+str(tune))
            self.agamma.append(tune)
            
            print('[gamma hat]\t'+str(gammahat))


            self.agammahat.append(gammahat)

            self.aenergymax.append(energymax)
            self.aenergymin.append(energymin)
            self.aenergypre.append(energynext)



        
        # print(pos_correction.shape)#partnum 3
        # print((self.gravity_vec*dt_frame/2).shape)#3
        # print((pos_correction/dt_frame+ _vel + self.gravity_vec*dt_frame/2).shape)#partnum 3

        # print('[vel mean]')
        # print(np.mean(_pos_correction1))
        # print(np.mean(_pos_correction2))
        # print(np.mean(_pos_correction))



        
        #zxc 先矫正位置，然后反推速度
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        
        print('--------------------------------')


        return pos2_corrected, vel2_corrected

    def call(self, inputs, fixed_radius_search_hash_table=None):
        #zxc 前向过程
        
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.

        zxc
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs

        _vel=vel    #is numpy
        partnum=pos.shape[0]
        if(not prm_train and not (prm_exportgap>1000) ):
            energy=getEnergy(vel=_vel,mask=self.mask,partnum=partnum)
            self.aenergy.append(energy)
            from util import ensure_2d
          
            print('[energy]\t'+str(energy))
            print(type(_vel))
            print(_vel.shape)#partnum 3
            mv=tf.reduce_sum(_vel,axis=0)
            print(mv.shape)#3
            print(mv.numpy().shape)
            
            self.amv.append(ensure_2d(mv.numpy()))
            print(ensure_2d(mv.numpy()).shape)
            # if(_vel.shape[0]>1):
                # assert(False)
            print('[mv x]'+str(mv[0]))
            print('[mv y]'+str(mv[1]))
            print('[mv z]'+str(mv[2]))

            


        #testterm toomuch
        # if(energy>15):
        #     print('too much energy')
        #     exit(0)

        #zxc 简单施加重力后的结果
        pos2, vel2 = self.integrate_pos_vel(pos, vel)

        #zxc 仅这一步用nn

        stm=time.time()
        
        pos_correction = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        
        self.infertime+=(time.time()-stm)
        
        # self.adelta_energy.append(np.sum(getDeltaEnergy(v=vel2,dv=pos_correction/self.timestep))/partnum)
        # print('[delta Energy]')
        # print(self.adelta_energy[-1])

        if(not prm_train and not (prm_exportgap>1000) ):
            gravity_mat=np.tile(np.array([0,-9.81,0]),(partnum,1))
        # self.adelta_energy2.append(np.sum(getDeltaEnergy2(
        #             dt=self.timestep,
        #             dx=pos_correction,
        #             v=vel,
        #             g=gravity_mat)
        #         )/partnum)
        # print(self.adelta_energy2[-1])
    


        #zxc 先矫正位置，然后反推速度
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        return pos2_corrected, vel2_corrected

    def init(self, feats_shape=None):
        """Runs the network with dummy data to initialize the shape of all variables"""
        pos = np.zeros(shape=(1, 3), dtype=np.float32)
        vel = np.zeros(shape=(1, 3), dtype=np.float32)
        if feats_shape is None:
            feats = None
        else:
            feats = np.zeros(shape=feats_shape, dtype=np.float32)
        box = np.zeros(shape=(1, 3), dtype=np.float32)
        box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

        _ = self.__call__((pos, vel, feats, box, box_feats))
