#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
import argparse
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_train, read_data_val
from collections import namedtuple
from glob import glob
import time
import tensorflow as tf
from prms import prmcontinueModel,continueModelname



# 会改变loss走向
tf.random.set_seed(5678)
tf.compat.v1.set_random_seed(5678)
np.random.seed(5678)

# print('\n\n\n[train]\n\n\n')
# xx=tf.random.normal((1,1))
# print(xx)


from utils.deeplearningutilities.tf import Trainer, MyCheckpointManager
from evaluate_network import evaluate_tf as evaluate
#zxc  ie evaluating...(validate ds)

import json



_k = 1000
#zxc max_iter
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(50 * _k, 0.001, 16)



#prm-------------------
#如果是涡度数据，就忽略yaml
prm_movecontainer=0
# 训练过程中移动容器

bvor=1
bdebug=0


storetensor=1
storelist=1


cut_thres=4.0


jsoname="lowfluid7.json"
# jsoname="lowfluid4.json"
# jsoname="lowfluid5.json"
jsoname="default.json"
jsoname="multi-10.json"
jsoname="temp.json"
jsoname="multi-10dense.json"
jsoname="lowfluidS.json"
jsoname="lowfluidS39.json"
jsoname="lowfluidS100.json"
jsoname="lowfluidS100cut.json"
jsoname="cc40.json"
jsoname="cc100.json"
jsoname="cc100less.json"
jsoname="csm40.json"
jsoname="tempcsm_mp100.json"
jsoname="csm_mp300.json"
jsoname="csm_mp300g3.json"
jsoname="csm300.json"
jsoname="temp.json"
# jsoname="csm_df300.json"






prm_cconvSceneConfig=1
# jsoname="lowfluid2cut.json"
dt_frame=0.016
# dt_frame=0.004




#not prm
iterall=0#全部遍历了一次
yamlname="error"

frameid=0
framep=0
segnum=0

bcutsparse=0
bevaluate=0
mycnt=0
if(bvor):
    bevaluate=0
else:
    bevaluate=1

sceneidx=0




if(bvor):
    #json存储训练超参数
    #yaml用于编号是否是之前训练过的场景。用于恢复训练。里面的数据路径不算数


    with open(jsoname, 'r') as file:
        jsondata = json.load(file)# inf json里如果写不存在的路径，不会自动新建
    print('['+jsoname+']')   
    _k                 =jsondata['_k']
    left=jsondata['left']
    right=jsondata['right']
    dtdir=jsondata['dtdir']
    gap=jsondata['gap']
    bcutsparse=jsondata['cutsparse']

    bmultiscene=jsondata['multiscene']
    scenenum=jsondata['scenenum']
    basescene=jsondata['basescene']
    jumpscene=jsondata['jumpscene']

    if(bdebug):
        print(jumpscene)
        print(len(jumpscene))
        exit(0)

    train_params = TrainParams(
        jsondata['max_iter'],
        jsondata['base_lr'],
        jsondata['batch_size'], 
        )


    frameid=left


# prm
# 每隔_k，evaluate一次.  _k会打印在训练时间左边


def create_model(**kwargs):
    from models.default_tf import MyParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork(**kwargs)
    return model
def get1ply(filename):
    from plyfile import PlyData
    import os

    #know 没找到文件会报错
    plydata = PlyData.read(filename)

    vertex =  plydata ['vertex']
       
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']


    combined = np.stack((x, y, z), axis=-1)
    return combined




pos0dict={}
pos1dict={}
pos2dict={}

boxpdict={}
boxndict={}

pos0dict[0]={}
pos1dict[0]={}
pos2dict[0]={}


pos0list=[[]]
pos1list=[[]]
pos2list=[[]]

boxplist=[]
boxnlist=[]




def mynext():#know

#inf FROM PINN-TORCH
    from plyfile import PlyData
    import os
    global dtdir
    global gap
    global frameid,framep
    global train_params
    global bmultiscene,sceneidx,scenenum
    global mycnt
    global iterall,segnum
    global prm_cconvSceneConfig
    global jumpscene




    batch={}
    batch['pos0']=[]
    batch['pos1']=[]
    batch['pos2']=[]
    batch['vel0']=[] 
    batch['box']=[]
    batch['box_normals']=[]

    for j in range(0,train_params.batch_size):#prm
        if(iterall):#已经遍历过整个dt一次，就不要重新读文件了

            if(storelist):
                pos0=pos0list[sceneidx][framep]
                pos1=pos1list[sceneidx][framep]
                pos2=pos2list[sceneidx][framep]


                boxp=boxplist[sceneidx]
                boxn=boxnlist[sceneidx]

            else:
                pos0=pos0dict[sceneidx][frameid]
                pos1=pos1dict[sceneidx][frameid]
                pos2=pos2dict[sceneidx][frameid]


                boxp=boxpdict[sceneidx]
                boxn=boxndict[sceneidx]

        # print('sample '+str(frameid))
        else:
            posname=dtdir+basescene+"-r"+str(sceneidx)+"_output/particle_object_0_"
            velname=dtdir+basescene+"-r"+str(sceneidx)+"_output/velocity_object_0_"
            
            if(prm_cconvSceneConfig):#csm
                if(iterall==0):
                    print('[reading file...],sceneidx='+str(sceneidx))
                posname=dtdir+basescene+str(sceneidx+1)+"_output/particle_object_0_"
                velname=dtdir+basescene+str(sceneidx+1)+"_output/velocity_object_0_"
                # containerposname=dtdir+basescene+str(sceneidx+1)+"_output/containerpos.npy"

            print('frame 1st ='+str(frameid))
            pos0=get1ply(posname+f"{frameid+0}.ply")
            pos1=get1ply(posname+f"{frameid+1}.ply")
            pos2=get1ply(posname+f"{frameid+2}.ply")

        
            # pos0=get1ply(posname+"{0:06}.ply".format(frameid+0))
            # pos1=get1ply(posname+"{0:06}.ply".format(frameid+1))
            # pos2=get1ply(posname+"{0:06}.ply".format(frameid+2))

            if(prm_movecontainer):
                containerpos=np.load(containerposname)

            if(prm_cconvSceneConfig==0):
                boxp=np.load(dtdir+"bp-"+basescene+"-r"+str(sceneidx)+".npy")
                boxn=np.load(dtdir+"bn-"+basescene+"-r"+str(sceneidx)+".npy")
            else:

                #csm
                cconvboxid=0
                with open("../datasets/csm/sim_{0:04d}/scene.json".format(sceneidx+1), "r") as f:
                    cconvboxid = json.load(f)["RigidBodies"][0]["boxid"]
                boxp=np.load("../datasets/Box_"+ str(cconvboxid)+".npy")
                boxn=np.load("../datasets/BoxN_"+str(cconvboxid)+".npy")


                if(prm_movecontainer):
                    print(containerpos.shape)#1000 3
                    print(frameid)#start from 1
                    print(containerpos[0])
                    print(sceneidx)#0
                    boxp+=containerpos[0]* ((frameid-1)*4+1)
                    # [0] 1 2 3
                    # [4]

                    # assert(False)

                
                
            segnum+=1


            if(storetensor):
                pos0=tf.convert_to_tensor(pos0)
                pos1=tf.convert_to_tensor(pos1)
                pos2=tf.convert_to_tensor(pos2)

                boxp=tf.convert_to_tensor(boxp)
                boxn=tf.convert_to_tensor(boxn)

            if(storelist):
                pos0list[sceneidx].append(pos0)
                pos1list[sceneidx].append(pos1)
                pos2list[sceneidx].append(pos2)

                if(frameid==left):
                    boxplist.append(boxp)
                    boxnlist.append(boxn)
            else:
                pos0dict[sceneidx][frameid]=pos0
                pos1dict[sceneidx][frameid]=pos1
                pos2dict[sceneidx][frameid]=pos2

                boxpdict[sceneidx]=boxp
                boxndict[sceneidx]=boxn


        batch['pos0'].append(pos0)
        batch['pos1'].append(pos1)
        batch['pos2'].append(pos2)

        batch['box'].        append(boxp)
        batch['box_normals'].append(boxn)

        # batch['vel0'].append(get1ply(velname+f"{frameid}.ply"))
        batch['vel0'].append((pos1-pos0)/dt_frame)

        # print('tensor vel0')
        # print(batch['vel0'][0].shape)
        # print(type(batch['vel0'][0]))
        # exit(0)


        if(bdebug):
            if(frameid==left):
                print('[boxp shape]')
                print(boxp.shape)
                print(boxn.shape)

            if(sceneidx==2):
                exit(0)
        

        frameid+=gap
        framep+=1

        if(frameid>right or frameid+2>right):#一个场景遍历完了
            frameid=left
            framep=0

            if(bmultiscene):
                sceneidx+=1
                # print('[posname]')
                # print(posname)
                #缺失场景 prm

                while(sceneidx in jumpscene):
                    sceneidx+=1

                    if(sceneidx-1 in jumpscene):
                        if(storelist):
                            if(iterall==0):
                                pos0list.append([])
                                pos1list.append([])
                                pos2list.append([])

                                boxplist.append([])
                                boxnlist.append([])
                
                if(iterall==0):
                    print('[sceneidx '+str(sceneidx)+']')
                
                if(sceneidx>=scenenum):#全部遍历完了
                    
                    sceneidx=0

              
                    if(iterall==0):
                        print('<all dt>')
                        print('seg num')
                        print(segnum)
                        mycnt+=1
                        print('sceneidx'+str(sceneidx))
                        print('frameid'+str(frameid))
                        print('framep\t'+str(framep))

                        print(len(pos0list))
                        print(pos0list[0][0].shape)
                        print(len(boxplist))
                        print(boxplist[0].shape)

                        iterall=1
                 

                else:
                    if(iterall==0):
                        pos0dict[sceneidx]={}
                        pos1dict[sceneidx]={}
                        pos2dict[sceneidx]={}

                        pos0list.append([])
                        pos1list.append([])
                        pos2list.append([])

                if(bdebug):
                    print('<scene '+str(sceneidx)+'>')


    # print('<dt loaded>')  

    # print(batch['vel0'][2].shape)
    # print(batch['pos0'][2].shape)

    return batch



def main():
    parser = argparse.ArgumentParser(description="Training script")#know
    parser.add_argument("cfg",
                        type=str,
                        help="The path to the yaml config file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
        yamlname=args.cfg.split(".")[0]
        if(yamlname!=jsoname.split(".")[0]):
            # 强制json和yaml使用相同的名字
            print("[Err]\t yaml json not match")
            exit(0)
    # the train dir stores all checkpoints and summaries. The dir name is the name of this file combined with the name of the config file
    train_dir = os.path.splitext(
        os.path.basename(__file__))[0] + '_' + os.path.splitext(
            os.path.basename(args.cfg))[0]
    if(bvor==0):
        val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))
        train_files = sorted(
            glob(os.path.join(cfg['dataset_dir'], 'train', '*.zst')))
            #zxc
        val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

        dataset = read_data_train(files=train_files,
                                batch_size=train_params.batch_size,
                                window=3,
                                num_workers=2,
                                **cfg.get('train_data', {}))
        # print('zxc dataset')
        # print(type(dataset))
        data_iter = iter(dataset)

    # print(type(data_iter))
    # print(data_iter.shape)
    # exit(0)
    trainer = Trainer(train_dir)

    model = create_model(**cfg.get('model', {}))
    model.init()
    if(prmcontinueModel):
        print("[continue on Model]")
        model.load_weights(continueModelname,by_name=True)




    boundaries = [
        25 * _k,
        30 * _k,
        35 * _k,
        40 * _k,
        45 * _k,
    ]
    lr_values = [
        train_params.base_lr * 1.0,
        train_params.base_lr * 0.5,
        train_params.base_lr * 0.25,
        train_params.base_lr * 0.125,
        train_params.base_lr * 0.5 * 0.125,
        train_params.base_lr * 0.25 * 0.125,
    ]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, lr_values)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn,
                                         epsilon=1e-6)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                     model=model,
                                     optimizer=optimizer)
                                     #zxc 恢复上次训练进度

    manager = MyCheckpointManager(checkpoint,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))

    def euclidean_distance(a, b, epsilon=1e-9):
        return tf.sqrt(tf.reduce_sum((a - b)**2, axis=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = tf.exp(-neighbor_scale * num_fluid_neighbors)
        if(bcutsparse):
            cut=1/(1+tf.exp(-9.0*(num_fluid_neighbors-cut_thres)))
            importance=importance*cut
        #prm。邻居粒子数量多时权重比较小。

        return tf.reduce_mean(importance *
                              euclidean_distance(pr_pos, gt_pos)**gamma)
                              #根据距离制定权重。
                              #距离L2。
                              

    @tf.function(experimental_relax_shapes=True)
    def train(model, batch):
        with tf.GradientTape() as tape:
            losses = []

            batch_size = train_params.batch_size
            for batch_i in range(batch_size):
                inputs = ([
                    batch['pos0'][batch_i], batch['vel0'][batch_i], None,
                    batch['box'][batch_i], batch['box_normals'][batch_i]
                ])

                pr_pos1, pr_vel1 = model(inputs)
                #know zxc 使用call()

                l = 0.5 * loss_fn(pr_pos1, batch['pos1'][batch_i],
                                  model.num_fluid_neighbors)
                                  #只有位置信息参与loss计算

                inputs = (pr_pos1, pr_vel1, None, batch['box'][batch_i],
                          batch['box_normals'][batch_i])
                          #zxc 只是将上述input中做了替换

                pr_pos2, pr_vel2 = model(inputs)

                l += 0.5 * loss_fn(pr_pos2, batch['pos2'][batch_i],
                                   model.num_fluid_neighbors)
                losses.append(l)

            losses.extend(model.losses)
            total_loss = 128 * tf.add_n(losses) / batch_size

            if(bdebug):
                tf.print(total_loss)
                print(total_loss)

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        checkpoint.restore(manager.latest_checkpoint)
        print('[restore]')
        #zxc know
    
    else:
        print('[new model]')

    display_str_list = []
    while trainer.keep_training(checkpoint.step,
                                train_params.max_iter,
                                #zxc

                                checkpoint_manager=manager,
                                display_str_list=display_str_list):

        data_fetch_start = time.time()
        if(bvor):
            batch=mynext()
        else:
            batch = next(data_iter)


        #zxc 取出一个batch。一个batch回传1次。一个batch是16帧数据。
        # print('-------------------------zxc batch next----------------')
        # print(type(batch))#dict

        # print(len(batch['pos0']))#16
        # print(batch['pos0'][0].shape)#13460 3
        # print(type(batch['pos0'][0]))#numpy
        # for i in range(16):
        #     print(batch['pos0'][i].shape)#N 3 N在变化


        # print(len(batch['pos1']))#16
        # print(batch['pos1'][0].shape)#13460 3


        # print(len(batch['pos2']))#16
        # print(batch['pos2'][0].shape)#13460 3
        # for i in range(16):
        #     print(batch['pos2'][i].shape)#N 3 


        # print(len(batch['vel0']))#16
        # print(batch['vel0'][0].shape)#13460 3
        # for i in range(16):
        #     print(batch['vel0'][i].shape)
        #     #N 3 N在变化，变化范围和上述完全相同，并且每一个batch的数据都不相同


        # print(len(batch['box']))#16
        # print(batch['box'][0].shape)#37005 3

        # print(len(batch['box_normals']))#16
        # print(batch['box_normals'][0].shape)#37005 3
        # zxccnt+=1
        # if(zxccnt==3):
        #     exit(0)
        
        batch_tf = {}
        if(storetensor):
                batch_tf=batch
        else:
            for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
                batch_tf[k] = [tf.convert_to_tensor(x) for x in batch[k]]
        #know zxc 全部转成tf tensor
        
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train(model, batch_tf)
        display_str_list = ['loss', float(current_loss)]
        #zxc

        if trainer.current_step % 10 == 0:
            with trainer.summary_writer.as_default():
                tf.summary.scalar('TotalLoss', current_loss)
                tf.summary.scalar('LearningRate',
                                  optimizer.lr(trainer.current_step))

        if trainer.current_step % (1 * _k) == 0:
            #zxc

            if(bevaluate):#swi
                for k, v in evaluate(model,
                                    val_dataset,
                                    frame_skip=20,
                                    **cfg.get('evaluation', {})).items():
                    with trainer.summary_writer.as_default():
                        tf.summary.scalar('eval/' + k, v)
    print('[saveh5]')
    model.save_weights(yamlname+'.h5')
    #zxc

    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
