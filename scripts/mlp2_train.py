import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.optimizers import Adam  
import numpy as np  
from prms import *
tf.random.set_seed(1234)
np.random.seed(1234)



prm_load=0
prmTrainMLP = prm_mix
assert(prmTrainMLP)
prm_save=1



#MLP解决线性规划问题：

# 训练集：(d1,d2,d3,d4,tune)  d被归一化为0~1，tune in [0,1] 
# 网络输出  a_i
#损失：  || Σ (d_i * a_i) - tune ||

#逐行归一化
def rownormal(X_train):
    for i in range(X_train.shape[0]):  
        X_train[i,:4]=      (X_train[i,:4]-np.min(X_train[i,:4]))/\
                    (np.max(X_train[i,:4])-np.min(X_train[i,:4]))
    return X_train
  
# 假设输入数据有5个特征，输出有4个值  
input_shape = (5,)  
output_dim = 5

samplenum=1
samplenum=5
samplenum=12
samplenum=24
samplenum=48
samplenum=100
# samplenum=3000

epochs=10
epochs=100
epochs=300
epochs=1000
# epochs=5000


def ones_initializer(shape, dtype=None):  
    return tf.ones(shape, dtype=dtype)  

# 生成一些随机数据作为示例（在实际应用中，你应该使用真实数据）  
X_train = np.random.random((samplenum*10, 5))  # 48个样本（3个epoch * 16个批次/epoch），每个样本5个特征  



X_train=rownormal(X_train)


for i in range(X_train.shape[0]):

    #tune只有2个值
    # if(X_train[i,4]>0.5):
    #     X_train[i,4]=10
    # else:
    #     X_train[i,4]=0

    pass
    # X_train[i,4]=X_train[i,4]*5

    # X_train[i,4]=X_train[i,4]*20-10

    


X_train=X_train.astype(np.float32)



# y_train = np.random.random((samplenum*10, 4))  # 对应的48个目标值，每个值4个输出  
y_train=tf.convert_to_tensor(X_train)

cnt=0

def custom_loss(y_true, y_pred):#已经按batch划分了
    print(y_pred.shape)#batch_size * 4
    # assert(False)
    dot_product=tf.reduce_sum(y_pred[:,:4] * y_true[:,:4],axis=1)
    print(dot_product.shape)
    print(dot_product)

   
    # assert(False)

    # maxe=np.amax(X_train[...,:4],axis=1)
    # mine=np.amin(X_train[...,:4],axis=1)
    maxe=1
    mine=0
    # print(maxe)
    # print(mine)
    # assert(False)
    tune=y_true[...,4]
    # return 1
    # print((((maxe-mine)*tune+mine-dot_product)/tune)**2)
    # assert(False)
    
    return tf.reduce_sum(((maxe-mine)*tune+mine-dot_product)**2)/batch_size

if(prm_load):
    tf.keras.models.load_model("./LinearMix.h5")
# 构建模型  
else:
    model = Sequential([  
        Dense(64,activation='sigmoid',use_bias=True,  input_shape=input_shape), 
        Dense(32,activation='sigmoid',use_bias=True),   
        # Dense(16,activation='relu',use_bias=True),                               
        # Dense(16,activation='relu',use_bias=True),    


        # Dense(output_dim,activation='softmax')#quanti__mix__
        # 5e-5
        # output_dim=5 能量后期会爆炸
        # output_dim=4 效果不行

        Dense(output_dim,activation='sigmoid')
        #5e-6
        #output_dim=5

        # Dense(output_dim,kernel_initializer=ones_initializer,activation='softmax') 
        #由于网络总是倾向于选择能量最低的模型 
        #2e-5




        # Dense(output_dim,activation='relu') 

        # Dense(output_dim,activation='sigmoid')  

    ])  

    # 编译模型，这里我们假设是一个回归问题，使用均方误差（MSE）作为损失函数  
    model.compile(optimizer=Adam(learning_rate=2e-3),  
                loss=custom_loss)  
    
  
# model.summary()
# exit(0)

  
# 训练模型  
# 注意：这里的steps_per_epoch和epochs参数是可选的，因为当你直接传递X_train和y_train时，  
# Keras会根据X_train的形状自动推断出批次大小和epoch数量（如果批次大小是已知的）。  
# 但是，为了明确起见，我们可以在这里设置它们。  
# 由于我们每个批次送入16个样本，总共有48个样本，所以steps_per_epoch=48//16=3。  
# 但是，由于Keras在fit方法中可以自动处理这个问题（当batch_size未明确设置时，它默认为32或数据集大小的最小值），  
# 我们实际上可以省略steps_per_epoch和batch_size参数，除非我们有特定的需求。  
# 然而，为了教育目的，我们将在这里明确设置它们。  
batch_size = 32  
if(prmTrainMLP):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)  
  
# 注意：上面的代码实际上设置了每个epoch有3个批次（因为48个样本/16个批次/epoch=3），  
# 并且我们训练了3个epoch，所以总共训练了9个批次（3个epoch * 3个批次/epoch）。  
  
# 如果你想查看模型的总结信息  
model.summary()
for i in range(0,6):
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<test>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
    x_test=np.random.random((3, 5))

    x_test=rownormal(x_test)


    if(i==1):
        x_test[:,4]=0
    elif(i==2):
        x_test[:,4]=1
    if(i==3):
        x_test[:,4]=0.5
    if(i==4):
        x_test=np.random.random((21, 5))
        x_test=rownormal(x_test)
        x_test[0,4]=0
        x_test[1,4]=0.1
        x_test[2,4]=0.2
        x_test[3,4]=0.3
        x_test[4,4]=0.4
        x_test[5,4]=0.5
        x_test[6,4]=0.6
        x_test[7,4]=0.7
        x_test[8,4]=0.8
        x_test[9,4]=0.9
        x_test[10,4]=1.0
        x_test[11,4]=1.1
        x_test[12,4]=1.2
        x_test[13,4]=1.3
        x_test[14,4]=1.4
        x_test[15,4]=1.5
        x_test[16,4]=1.6
        x_test[17,4]=1.7
        x_test[18,4]=1.8
        x_test[19,4]=1.9
        x_test[20,4]=2.0




    tune=x_test[...,4]
    maxe=np.amax(x_test[...,:4],axis=1)
    mine=np.amin(x_test[...,:4],axis=1)
    maxe=1
    mine=0

    print('model energy--------------')
    print(x_test)

    pre=model.predict(x_test)
    pre=pre[:,:4]
    # if(i==5):
        
    #         # 假设你有一个数组  
    #         arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])  

    #         # 创建一个与原始数组形状相同的新数组，并初始化为0  
    #         result = np.zeros_like(x_test[:,:4])  
            
    #         # 使用numpy.argmax找出最大值的索引  
    #         for j in range(x_test.shape[0]):
    #             max_index = np.argmax(x_test[j,:4])
            

            
    #         # 在最大值的索引位置设为1  
    #             result[j,max_index] = 1  
            
    #         pre=result

    print('--------------coff')
    print(pre)
    print('--------------pre strength:')

    pre_strength=(maxe-mine)*np.sum(x_test[...,:4]*pre,axis=1)+mine
    print(pre_strength)

    print('--------------expected strength')


    print((maxe-mine)*tune+mine)

    print('------------------loss ')
    print((pre_strength-tune)**2)

if(prm_save):
    model.save("./LinearMix.h5")

def predictincconv(d1,d2,d3,d4,partnum,tune):
    
    x_test=np.array([d1,d2,d3,d4])
    # print(x_test.shape)
    # print(x_test)
    x_test=(x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
    x_test=np.concatenate([x_test,np.array([tune])])
    print('[normalized]')
    print(x_test)
    x_test=np.array([x_test])
    print(x_test.shape)

    # assert(False)
    pre=model.predict(x_test)

    return pre