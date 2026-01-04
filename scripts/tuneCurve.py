def x2(tune0,tuneend,step,num_steps):
    

    return ((num_steps-step)**2) * (tune0-tuneend)/(num_steps**2) + tuneend


def x1(tune0,tuneend,step,num_steps):
    

    return ((num_steps-step)**1) * (tune0-tuneend)/(num_steps**1) + tuneend


import math

def oscillating(t, y0, y1, t0, n=4):
    A = (y0 - y1) / 2
    C = (y0 + y1) / 2
    omega = 2 * math.pi * n / t0
    return A * math.sin(omega * t) + C


# t = 1.0   # 计算时间点
# y0 = 10   # 最大值
# y1 = 2    # 最小值
# t0 = 5    # 震荡周期范围
# n = 3     # 震荡次数
