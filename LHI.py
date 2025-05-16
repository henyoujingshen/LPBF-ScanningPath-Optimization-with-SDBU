import math
#以下是二维的LHI,生成次序可以接draw_strategy可视化

# 定义一个函数来计算两点之间的欧几里得距离
# def euclidean_distance(point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2
#     return math.sqrt((x2-x1)**2 + (y2-y1)**2)
#
# # 定义一个函数来查找下一个数字的位置
# def next_position(matrix, current_pos):
#     max_distance = -1  # 最大距离初始化为-1
#     next_pos = None  # 下一个位置初始化为None
#
#     # 对于每个位置，计算其与当前位置的距离并找到最远的位置
#     for i in range(len(matrix)):
#         for j in range(len(matrix[i])):
#             if matrix[i][j] == 0:
#                 distance = euclidean_distance(current_pos, (i, j))
#                 if distance > max_distance:
#                     max_distance = distance
#                     next_pos = (i, j)
#
#     return next_pos
#
# # 初始化5x5矩阵
# matrix = [[0 for j in range(5)] for i in range(5)]
#
# # 从(0,0)位置开始填充矩阵
# current_pos = (0, 0)
# for num in range(1, 26):
#     matrix[current_pos[0]][current_pos[1]] = num
#     current_pos = next_position(matrix, current_pos)
#
# # 打印填好的矩阵
# for row in matrix:
#     print(row)

#以下是一维的LHI
import math

# 定义一个函数来计算两点之间的欧几里得距离（一维情况）
def euclidean_distance_1d(point1, point2):
    return math.sqrt((point2 - point1) ** 2)

# 定义一个函数来查找下一个数字的位置（一维情况）
def next_position_1d(vector, current_pos):
    max_distance = -1  # 最大距离初始化为-1
    next_pos = None  # 下一个位置初始化为None

    # 对于每个位置，计算其与当前位置的距离并找到最远的位置
    for i in range(len(vector)):
        if vector[i] == 0:
            distance = euclidean_distance_1d(current_pos, i)
            if distance > max_distance:
                max_distance = distance
                next_pos = i

    return next_pos

# 初始化1x50向量
vector = [0 for i in range(50)]

# 从0位置开始填充向量
current_pos = 0
for num in range(1, 51):
    vector[current_pos] = num
    current_pos = next_position_1d(vector, current_pos)

# 打印填好的向量
print(vector)
