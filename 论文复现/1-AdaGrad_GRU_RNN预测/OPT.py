def count_valid_solutions(n, min_interval, max_interval, last_start, last_end):
    dp = [0] * n
    # 初始化第一个1的位置在min_interval+1到max_interval+1之间
    for k in range(min_interval + 1, max_interval + 2):
        dp[k] = 1
    # 填充动态规划数组
    for k in range(min_interval + 1 + 1, n):
        for j in range(max(k - (max_interval + 1), min_interval + 1), k - (min_interval + 1) + 1):
            if 0 <= j < n:
                dp[k] += dp[j]
    # 计算最后1在last_start到last_end之间的解的数量
    total = sum(dp[last_start:last_end + 1])
    return total

# 参数设置
n = 96
min_interval = 11  # 对应11个0,距离为12
max_interval = 15  # 对应15个0,距离为16
last_start = 84
last_end = 95

# 计算解的数量
total_solutions = count_valid_solutions(n, min_interval, max_interval, last_start, last_end)
print(f"Total number of valid solutions: {total_solutions}")

import gurobipy as gp
from gurobipy import GRB

# 参数设置 interval 代表1之间间隔的0的数量
min_interval = 11    # 对应3h的情况
max_interval = 15    # 对应4h的情况
n = 96

# 创建模型
model = gp.Model('optimization_model')

# 创建变量
x = model.addVars(n, vtype=GRB.BINARY, name='x')  # x[0] 到 x[95] 分别代表96个位置是否为1

# 约束条件

# 条件1: 前 min_interval 个位置必须是0
for i in range(min_interval):
    model.addConstr(x[i] == 0)

# 条件2: 在 min_interval 到 max_interval 之间必须有一个1
model.addConstr(gp.quicksum(x[i] for i in range(min_interval, max_interval)) == 1)

# 条件3a: 两个1之间的最小距离至少是 min_interval
for i in range(n):
    for j in range(i + 1, i + min_interval):
        if j > n-1:
            break
        model.addConstr(x[i] + x[j] <= 1)

# 条件3b: 两个1之间的最大距离不超过 max_interval
for i in range(n - max_interval):
    model.addConstr(gp.quicksum(x[j] for j in range(i, i + max_interval)) >= 1)

# 目标函数: 设为0,因为我们不关心目标值,只寻找可行解
model.setObjective(0, GRB.MAXIMIZE)

# 设置求解参数
model.Params.PoolSearchMode = 2  # 寻找所有解决方案
model.Params.PoolSolutions = 10000  # 设置一个足够大的上限
model.Params.PoolGap = 0.0  # 找到所有解决方案,不考虑目标值差距

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print('Number of solutions found: ', model.solCount)
    # for sol in range(model.solCount):
    #     model.setParam(GRB.Param.SolutionNumber, sol)
    #     solution = [x[i].Xn for i in range(n)]
    #     print('Solution ', sol + 1, ':', end=' ')
    #     for i in range(n):
    #         if solution[i] == 1:
    #             print(i + 1, end=' ')
    #     print()
else:
    print('No feasible solution found.')
# 输出结果
print('12: 2:45~3:00  16: 3:45~4:00 20: 4:45~5:00 24: 5:45~6:00 28: 6:45~7:00 32: 7:45~8:00 36: 8:45~9:00 40: 9:45~10:00 44: 10:45~11:00 48: 11:45~12:00 52: 12:45~13:00 56: 13:45~14:00 60: 14:45~15:00 64: 15:45~16:00 68: 16:45~17:00 72: 17:45~18:00 76: 18:45~19:00 80: 19:45~20:00 84: 20:45~21:00 88: 21:45~22:00 92: 22:45~23:00 96: 23:45~24:00')
if model.status == GRB.OPTIMAL:
    print('Number of solutions found: ', model.solCount)
    for sol in range(model.solCount):
        model.setParam(GRB.Param.SolutionNumber, sol)
        solution = [x[i].Xn for i in range(n)]
        print('Solution ', sol + 1, ':', end=' ')
        for i in range(n):
            if solution[i] == 1:
                print(i + 1, end=' ')
        solution = [int(x[i].Xn) for i in range(n)]
        print()
        print(solution)
        print('=========================')
else:
    print('No feasible solution found.')