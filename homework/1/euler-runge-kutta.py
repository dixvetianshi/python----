import numpy as np
import matplotlib.pyplot as plt

# 使matplotlib支持中文显示
plt.rcParams["font.family"] = "SimHei"  
# 防止负号显示异常
plt.rcParams["axes.unicode_minus"] = False

# 封装 Euler 方法，参数 f 为右侧函数，y0 为初始值，t_values 为时间节点数组
def euler_solver(f, y0, t_values):
    y = np.zeros(len(t_values))
    y[0] = y0
    for i in range(len(t_values)-1):
        h = t_values[i+1] - t_values[i]
        y[i+1] = y[i] + h * f(t_values[i], y[i])
    return y

# 封装 RK4 方法（四阶 Runge-Kutta），参数说明同上
def rk4_solver(f, y0, t_values):
    y = np.zeros(len(t_values))
    y[0] = y0
    for i in range(len(t_values)-1):
        h = t_values[i+1] - t_values[i]
        t_i = t_values[i]
        k1 = f(t_i, y[i])
        k2 = f(t_i + h/2, y[i] + h/2 * k1)
        k3 = f(t_i + h/2, y[i] + h/2 * k2)
        k4 = f(t_i + h, y[i] + h * k3)
        y[i+1] = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return y

def main():
    # 参数设置
    a = 1.0       # 衰减系数
    y0 = 1.0      # 初始值 y(0)
    t0 = 0.0      # 起始时间
    t_end = 5.0   # 结束时间
    N = 100       # 时间步数
    t_values = np.linspace(t0, t_end, N+1)
    
    # 定义微分方程的右侧函数，可以通过闭包传递参数 a
    f = lambda t, y: -a * y

    # 调用函数分别计算 Euler 与 RK4 的数值解
    y_euler = euler_solver(f, y0, t_values)
    y_rk4   = rk4_solver(f, y0, t_values)
    
    # 计算精确解
    y_exact = y0 * np.exp(-a * t_values)

    # 绘图比较三者结果
    plt.figure(figsize=(8, 6))
    plt.plot(t_values, y_exact, 'r--', label='精确解')
    plt.plot(t_values, y_euler, 'bo-', label='Euler 数值解')
    plt.plot(t_values, y_rk4, 'gs-', label='RK4 数值解')
    plt.xlabel('时间 t')
    plt.ylabel('y')
    plt.title('一阶微分方程数值求解比较')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
