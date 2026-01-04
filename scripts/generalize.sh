num_steps=4
num_steps=20
# num_steps=50
# num_steps=100
num_steps=150
# num_steps=250
num_steps=400
num_steps=500
num_steps=1000
# num_steps=1200
# num_steps=1300
# num_steps=1500
# num_steps=2000
# num_steps=2300
num_steps=2800
# num_steps=4000


# modelname1=csm400_1111
# modelname2=csm400_1111
# modelname1=csm300
# modelname1=csm300_6789
# modelname1=csm200_1111
# modelname1=default
# modelname1=csm_df300_sms_1111
# modelname1=csm350_1111


#model sets B
# modelname1=csm_df300_1111
# modelname1=csm_mp300
modelname1=pretrained_model_weights
# modelname1=csm300_1111
# modelname1=skdf



# modelname1=stabledf
# modelname1=early__csm_mp300
# modelname1=dup_stable_csm_df300



#fe(filter extent)
# modelname1=fe5_csm_df300
# modelname1=fe7_csm_df300
# modelname1=fe13_csm_df300
# modelname1=fe5_noobs_csm_df300
# modelname1=fe13_noobs_csm_df300
# modelname1=fe13_noobs_dmcf_csm_df300
# modelname1=fe18_noobs_csm_df300





#dmcf
# modelname1=dmcf_csm_df300_1234
# modelname1=dmcf_csm_df300_5678
# modelname1=dmcf_csm_mp300_5678
# modelname1=dmcf_csm300_5678


# modelname1=RS6__csm_mp300
# modelname1=RSL3__csm_mp300


# modelname1=ContOnPretrained__csm_mp300





# scnename=example_static1.5x
# scnename=example_static_stick
# scnename=example_still__L
# scnename=example_still__2L
scnename=example_static
# scnename=example_quad
# scnename=mc_ball_2velx_0602_slow2
# scnename=mc_ball_2velx_0602_slow2_2
# scnename=mc_ball_2velx_0602
scnename=mc_ball_2velx_0602_wake
scnename=mc_ball_2velx_0602_wake_wide
# scnename=0711
# scnename=0602_emit
# scnename=0602_emit_q
# scnename=0602_emit_q_a
# scnename=0602_emit_thin
# scnename=0602_emit_thin_b
# scnename=emit_big
# scnename=emit_large
# scnename=emit_large_myemit
# scnename=emit_large_1layer
# scnename=0712
# scnename=high_fluid_mcvsph-fluid
# scnename=maxpartnum
# scnename=rotating_panel
# scnename=example_propeller
# scnename=watervessel
# scnename=watervessel1.4x
# scnename=watervessel1.5x
# scnename=watervessel1.13x
# scnename=watervessel1.06x
# scnename=watervessel0.8x
# scnename=watervessel0.7x
# scnename=watervessel0.9x
# scnename=wavetower
# scnename=wavetowerstatic
scnename=wavetowerstatic1.3x
# scnename=momeConse
# scnename=propeller
# scnename=propellerlarge
# scnename=streammultiobjs2x
# scnename=streammultiobjs3x
# scnename=streammultiobjs3xcut
# scnename=streammultiobjsHorizon
# scnename=streammultiobjs2xCut
# scnename=taylorvortex1.5scale
# scnename=rotatingpanelstatic
# scnename=rotatingpanelstatic1.5x
# scnename=rotatingpanelstatic2x_thin
scnename=rotatingpanelstatic2x
# scnename=propellerbiglarge2
# scnename=propeller2+large2









# testname=wallmove___
# testname=____1.5x___
# testname=____90w___nobox
testname=___temp___test__rs12___
#testname=___push680___
# testname=_____stop650__
# testname=____onlyrigid__
# testname=____horizon__
# testname=__ONLYNPZ_____
testname=__wait500__
testname=__wait__inf__
# testname=__cont_on_stable_400__
# testname=_____notower___
# testname=___80w__onlynpz___
testname=____nopanel___
# testname=____exportall___
testname=____cp_____
testname=____FullProcess___
# testname=____norotating___
testname=__Cont770__
# testname=_______
# testname=_____rmLeft__v0.75__


lize=lize




ckpt=_50k
syncdir=/w/cconv-dataset/sync/

filename=$lize$testname$modelname1$scnename

python prms.py --weights $modelname1.h5 \
                --type 1 \
                --scene $scnename.json || exit 1


rm -r $syncdir$filename
./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --output $syncdir$filename \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py



cd $syncdir
# rm $filename.zip
# zip -rq  $filename.zip $filename && rm -r $filename


cd /workspace/DeepLagrangianFluids-FORK/scripts/


# ./train_network_tf.py $modelname1.yaml


