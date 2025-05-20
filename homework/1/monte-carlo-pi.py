# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 使matplotlib支持中文显示
plt.rcParams["font.family"] = "SimHei"  
# 防止负号显示异常
plt.rcParams["axes.unicode_minus"] = False

def monte_carlo_pi(num_points, plot_sample_size=1000):
    """
    用 Monte Carlo 方法估计 π 值，同时返回用于绘图的随机点集合。
    
    参数:
        num_points - 随机生成的总点数（用于计算 π 值）
        plot_sample_size - 如果随机点总数大于此数，则抽样用于绘图的点数
    返回:
        pi_est - 估计的 π 值
        points_plot - 用于绘图的随机点数组 (shape: (plot_sample_size, 2))
        inside_plot - 布尔数组，表明每个用于绘图的点是否在单位圆内
    """
    # 在二维区域 [-1, 1] × [-1, 1] 内生成全部随机点
    points = np.random.uniform(-1, 1, (num_points, 2))
    
    # 计算每个点到原点 (0, 0) 的距离
    distances = np.linalg.norm(points, axis=1)
    
    # 判断哪些点落在单位圆（半径为 1）内
    inside = distances <= 1.0
    count_inside = np.sum(inside)
    
    # 根据面积比关系，估计 π 的值
    pi_est = 4 * count_inside / num_points

    # 为了防止绘图时点数过多导致卡顿，随机抽样用于绘图的点
    if num_points > plot_sample_size:
        indices = np.random.choice(num_points, plot_sample_size, replace=False)
        points_plot = points[indices]
        inside_plot = inside[indices]
    else:
        points_plot = points
        inside_plot = inside
    
    return pi_est, points_plot, inside_plot

def main():
    num_points = 100000000  # 例如生成 1,000,000 个随机点用于估计 π
    plot_sample_size = 5000  # 绘图时只显示 5000 个随机点
    pi_est, points_plot, inside_plot = monte_carlo_pi(num_points, plot_sample_size)
    print("Monte Carlo 估计的 π 值为:", pi_est)
    
    # 绘图展示
    plt.figure(figsize=(6, 6))
    
    # 分离圆内点和圆外点（基于抽样得到的随机点）
    inside_points = points_plot[inside_plot]
    outside_points = points_plot[~inside_plot]
    
    plt.scatter(inside_points[:, 0], inside_points[:, 1],
                color='blue', s=1, label='圆内点')
    plt.scatter(outside_points[:, 0], outside_points[:, 1],
                color='red', s=1, label='圆外点')
    
    # 绘制单位圆的边界
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    plt.plot(x_circle, y_circle, color='black', lw=2, label='单位圆')
    
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Monte Carlo 估计 π')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
