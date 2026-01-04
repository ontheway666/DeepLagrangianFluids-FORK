



# -------------------------

prmcontinueModel=0
# 继续在之前已有模型上训练，创建新文件
continueModelname="pretrained_model_weights.h5"


prmdmcf=0



prmsimplebc=0



prmcutbox=0



#在统计信息中过滤掉出界的粒子
prm_mask=0



#自行设置emit，忽略json里的水块信息
prm_myemit=1



prm_edit=0







prm_round=0
eps=4






prm_wallmove=1
prm_motion='still'
prm_motion='error'
movetime=-1



prm_maxenergy=0
prm_exactarg=0




prm_mix=0
prm_mixmodel="error"




prm_linear=0
prmtune0=0.3
prmtuneend=0.3
prmfixcoff=0




prm_mlpexact=0
prm_mlpexact_2=0
prm_needratio=      True
prm_energyratio=    True
ratio0=0.5
prm_3dim=           False
prm_customtune=0



prm_sus=0



prmrestrict_upward=0



prmmixstable=0
prmstableratio=0.95



prm_pointwise=0



prm_area=0



#仅输出初始场景的信息，通过generalish.sh使用
prm_outputInitScene=0


prm_continuerun=0
prm_simrigidwithoutfluid=1
# 在continuerun的情况下，运行之前的帧，让固体到达指定的位置，但是不模拟液体
prm_resumestep=770
#从这一帧开始续上




prm_train=0



prm_moveOutPart=0


prm_slopeGravity=0
prm_0gravity=0



prm_exportgap=1
# prm_exportgap=10
# prm_exportgap=50
# prm_exportgap=1000000
prmexportrig=1
prmexportallrigidperframe=1


prm_savelvel=0





prmFE=0
prmFEval=13
# 默认是9倍，调整为5,13,18,


# 可以自行调整参数，但是系统会再帮你调一次，避免出现错误，或者信任系统，自己不用调。


# 参数很多，需要校验，否则一旦一个输错了重新跑成本是很高的


# 当执行预测或者训练之前，先做校验.这里不能直接帮忙自动修改参数，
# 因为import时会重新导入前面的参数，但是import又不能执行校验，因为拿到shell里的东西很麻烦
# 所以只在前面修改
if __name__ == "__main__":
   

    import argparse
    localparser = argparse.ArgumentParser(description="check legal")
    localparser.add_argument(
            "--weights",
            type=str,
            required=True,
            help=
            "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
        )

    localparser.add_argument(
        "--type",
        type=int,
        required=True,
        help=
        "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )

    localparser.add_argument("--scene",
                        type=str,
                        required=False,
                        help="A json file which describes the scene.")
    localparser.add_argument("--prm_mix",
                        type=int,
                        required=False,
                        default=-1,
                        help="The device to use. Applies only for torch.")


    localargs = localparser.parse_args()
    localweights = localargs.weights
    localscene=localargs.scene
    localtype=localargs.type
    localmix = localargs.prm_mix

    if(localmix==1):
        assert(prm_mix==1)
    else:
        assert(prm_mix==0)


    if(localtype==1):
        assert(not prm_train)
        
    elif(localtype==2):
        assert(prm_train)
        
    else:
        assert(False)




    print(localweights)


    if("dmcf" in localweights):
        assert(prmdmcf)

    if(localweights.find("fe13")==0):
        assert(prmFE)
        assert(prmFEval==13)

    elif(localweights.find("fe18")==0):
        assert(prmFE)
        assert(prmFEval==18)
    elif(localweights.find("fe5")==0):
        assert(prmFE)
        assert(prmFEval==5)
    elif(localweights.find("fe7")==0):
        assert(prmFE)
        assert(prmFEval==7)
    else:
        assert(not prmFE)



    if(not prm_train):

        if("vessel" in localscene 
        or "wavetower" in localscene
        or "rotatingpanel"in localscene
        or "0602_wake" in localscene
        or "horizon" in localscene
        or "3x" in localscene):
            assert(prm_myemit)

        elif("example_static" in localscene
        or ("0602" in localscene and not "wake" in localscene)):
            assert(not prm_myemit)
        else:
            assert(False)


    if(prmFE and prmdmcf):
        assert("dmcf" in localweights 
        and ("fe5" in localweights or "fe13" in localweights or "fe18" in localweights or "fe7" in localweights))


assert(not(prmFE and prmcontinueModel))
assert(not(prmdmcf and prmcontinueModel))
