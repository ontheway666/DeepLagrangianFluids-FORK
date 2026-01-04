num_steps=3
num_steps=5
# num_steps=10
num_steps=250
# num_steps=400
num_steps=505
# num_steps=600
# num_steps=800
num_steps=1000
# num_steps=1100
# num_steps=1200
# num_steps=1400
num_steps=1500
# num_steps=1700
# num_steps=2000
# num_steps=2300
# num_steps=2500
num_steps=2800
# num_steps=5000



modelname1=csm_df300_1111




scnename=mc_ball_2velx_0602
# scnename=example_static
# scnename=emit_large
# scnename=emit_large_myemit
# scnename=emit_large_myemit_slope
# scnename=emit_large_myemit_slope_2
# scnename=emit_large_myemit_slope_3
# scnename=emit_large_myemit_slope_move
# scnename=emit_large_myemit_slope_move_x+
# scnename=example_long2z+2
# scnename=example_still__L
# scnename=example_still__2L
# scnename=example_propeller
# scnename=maxbox
# scnename=high_fluid_mcvsph-fluid
# scnename=rotating_panel
# scnename=watervessel
# scnename=watervessel1.13x
# scnename=wavetower
scnename=wavetowerstatic
# scnename=propellerlarge
# scnename=propellerlarge1.5x
# scnename=propeller2large
# scnename=propeller2large2
# scnename=propeller2large21.5x
# scnename=taylorvortex
# scnename=streammultiobjs3x
# scnename=streammultiobjs3xcut
# scnename=streammultiobjs3xcutbunny12
# scnename=streammultiobjs3xcutbunny12-1.5x
# scnename=streammultiobjs2x
# scnename=streammultiobjsHorizon
# scnename=streammultiobjs2xCut
# scnename=rotatingpanelstatic
# scnename=taylorvortex1.5scale
# scnename=propellerbiglarge2
# scnename=propeller2+large2
# scnename=propeller2+large2fill











testname=emax_b_
# testname=CP__emax_b_
# testname=emax_b_1.5x__
# testname=cp__emin_b_
# testname=cp__emax_b_
# testname=minmaxmin_b_
# testname=area_n10tn2
# testname=areamaxzt0_
# testname=area_n10tn2
# testname=emin__period0.75__
# testname=sus320__
# testname=emin__sus150_stable0.95__0.5rot__
# testname=emin__rot0.5__
# testname=emax__rot0.5__
# testname=emin__rot.125__
# testname=emin__acc___
testname=emin__
# testname=emin__static80__
# testname=emin__nopush__
# testname=horizon_emin__
# testname=emin__moveoutpart___
# testname=emax__restrict__acc___
# testname=ONLYNPZ__emin__
# testname=ONLYNPZ__q0.5___
# testname=quanti0.5____period0.6__startpush650___
# testname=emin__test___
# testname=maxmin__
# testname=emin__period0.5__
# testname=quanti0.5____acc___
# testname=quanti0.5____period0.75__wait500___
# testname=quanti0.5____period0.75__push680__
# testname=q0.5__0.2_line__
# testname=q0.5__0.2_osc__
# testname=q0.7__0.2__osc8__
# testname=q1.2__0.5__osc6__
# testname=q5__FullProcess___
# testname=quanti0.65____period0.75__
# testname=emin__rot0.25__holprop__
# testname=emin__rot0.25__
# testname=sus_stable__
# testname=emin_stable0.95__
# testname=emin_stable0.5__
# testname=emin__NoProp__
# testname=emin__rot0.5__restrictUp__
# testname=emin__1.5rot__
# testname=eminC__
# testname=emin__exactarg__
# testname=pw_max_ 
# testname=pw_min_
# testname=eminmaxmin_d7_
# testname=emin4_
# testname=pw_min4_
# testname=pw_max4_
# testname=___maxmin___
# testname=___min___
# testname=0602__switch2__
# testname=temp____
# testname=temp__slope1_v3__
# testname=quanti__1.15-300start__
# testname=quanti__1.15-clampn2__
# testname=quanti__cp_1.25curve__
# testname=quanti__tau1.25t0.7curve__
# testname=quanti__tau1.25t0.5line__
# testname=area_linear__myemit_
# testname=quanti__1.25line__
# testname=quanti__0.75____
# testname=quanti__0.5____
# testname=quanti__0.5t0.2line____
# testname=quanti__0.5t0.75line____
# testname=quanti__0.2t1line____
# testname=quanti__0.25____
# testname=quanti__0____
# testname=pw___min__
# testname=pw___max__
# testname=temp____random_linear___
# testname=temp__myeit__
# testname=area__emin1.2t0.5and0.5___
# testname=area__0.8t0.5and0.5___
# testname=area__1.2t0.2and0.5___
# testname=area__maxmin___
# testname=area__0.7and0.5__
# testname=ONLYNPZ__quanti__tau1.25t0.2curve__
# testname=ONLYNPZ__quanti_1.25_masked___
# testname=ONLYNPZ__quanti_0.625___
# testname=ONLYNPZ__emaxb__CP_
# testname=exact_0.75__
# testname=exact_0.8__
# testname=_emin___90w___nobox
# testname=fixcoff__q1.25t0.2__
testname=__Cont770__q3_p75__
# testname=__Cont770__emin__








ckpt=_50k
syncdir=/w/cconv-dataset/sync/
filename=$testname$modelname1$scnename
outputdir=$syncdir$filename

#同时加载多个模型，施加修正

python prms.py --weights $modelname1.h5 \
                --type 1 \
                --prm_mix 1\
                --scene $scnename.json || exit 1
                

# rm -r $outputdir
./run_network.py --weights $modelname1.h5 \
                --scene $scnename.json \
                --prm_mix 1\
                --prm_mixmodel $modelname2.h5 \
                --output $outputdir \
                --write-ply \
                --num_steps $num_steps \
                train_network_tf.py


cd $syncdir


# rm  $filename.zip
# zip -rq  $filename.zip $filename && rm -r $filename


cd /workspace/DeepLagrangianFluids-FORK/scripts/

