import numpy as np
import geatpy as ea
import time
from utils.evaluation import evaluator
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [1] # 初始化maxormins（目标最小化标记列表）
        Dim = 25 # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量类型列表），元素为0表示对应的变量是连续的；1表示是离散的；2表示是二进制的。
        lb = [1] * Dim # 决策变量下界
        ub = [25] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界是否包含在可行域中
        ubin = [1] * Dim # 决策变量上边界是否包含在可行域中
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen.astype(int)  # 得到决策变量矩阵，转换为整数类型

        f_n = np.zeros((pop.sizes, 1))  # 初始化f(n)矩阵

        for i in range(pop.sizes):  # 遍历每个个体

            x = Vars[i, :]  # 获取第i个个体的排列编码

            f_n[i] = self.ml_modeling(x)  # 调用机器学习

        pop.ObjV = f_n  # 把求得的目标函数值赋值给种群pop的ObjV属性

    def ml_modeling(self, x):
        """
        这里是有机器学习的定义，根据x计算f(n)，返回一个浮点数。
        """
        x=x.reshape(5,5)
        evaluate=evaluator(x,'rnnDNN')
        output=evaluate.calculate_output()
        return output

problem = MyProblem()       # 生成问题对象

"""=======================种群设置======================="""
Encoding ='P'              # 编码方式('P'表示采用排列编码)
NIND    = 100              # 种群规模(即个体数目)
Field   = ea.crtfld(Encoding,problem.varTypes,problem.ranges,
                     problem.borders)    	# 创建区域描述器(这里仅使用前两列数据)
population    = ea.Population(Encoding , Field , NIND)  	# 实例化种群对象(生成初代种群)

"""=======================算法参数设置====================="""
myAlgorithm = ea.soea_SEGA_templet(problem, population) # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 20 # 最大进化代数
myAlgorithm.drawing = 1 # 绘图方式，0表示不绘图，1表示绘制结果图
[BestIndi, population] = myAlgorithm.run() # 执行算法模板，得到最后一代种群以及执行时间等信息

"""==================================输出结果============"""
print('评价次数：%s' % myAlgorithm.evalsNum)
print('时间已过 %s 秒' % myAlgorithm.passTime)
if BestIndi.sizes != 0:
    print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
    print('最优的控制变量值为：')
    for i in range(BestIndi.Phen.shape[1]):
        print(BestIndi.Phen[0, i])
else:
    print('没找到可行解。')