modelname=stable_1-4_csm_df300


if [ -f "$modelname.h5" ]; then
    echo "文件 $modelname.h5 存在，脚本终止"
    exit 1
fi


python prms.py --weights $modelname.yaml \
                --type 2 || exit 1

./train_network_tf.py $modelname.yaml