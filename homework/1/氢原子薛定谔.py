import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 使matplotlib支持中文显示
plt.rcParams["font.family"] = "SimHei"  
# 防止负号显示异常
plt.rcParams["axes.unicode_minus"] = False

def solve_hydrogen_atom(l=0, N=1000, r_max=30.0):
    """
    数值求解氢原子径向薛定谔方程（经过变量替换 u(r)=r*R(r)）。
    参数：
      l    : 角动量量子数（此处示例取 l=0，即 s 轨道）
      N    : 离散网格的点数（包括边界）
      r_max: 半径上限（选择足够大使得 u(r_max)≈0）
    返回：
      r_in   : 内部网格点（排除 r=0 和 r=r_max）
      energies: 数值求解得到的能量本征值（升序排列）
      u_eig  : 对应的径向函数 u(r)（归一化满足 ∫|u(r)|²dr=1）
      R_eig  : 对应的径向波函数 R(r)=u(r)/r（归一化满足 ∫|R(r)|²r²dr=1）
    """
    # 为避免 r=0 奇异性，这里采用包含端点的均匀网格，然后剔除 r=0 和 r_max 点
    r = np.linspace(0, r_max, N)
    dr = r[1] - r[0]
    # 排除边界：注意这里 r[0]=0, r[-1]=r_max（波函数均设置为 0）
    r_in = r[1:-1]
    n_in = len(r_in)
    
    # 构造动能算符离散表示：-1/2 d²/dr²
    # 中心差分近似： u''(r_i) ≈ (u_{i+1}-2u_i+u_{i-1})/dr²
    # 则有 -1/2 u''(r_i) = +1/dr² * u_i - 1/(2dr²)*(u_{i+1}+u_{i-1])
    diag = np.full(n_in, 1.0/dr**2)
    off_diag = np.full(n_in - 1, -1.0/(2*dr**2))
    T = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    
    # 构造有效势：V_eff = -1/r + l(l+1)/(2r^2)
    V_eff = -1.0 / r_in
    if l != 0:
        V_eff += l*(l+1)/(2.0 * r_in**2)
    V = np.diag(V_eff)
    
    # 整个哈密顿量矩阵
    H = T + V
    
    # 求解特征值问题
    energies, u_eig = eigh(H)
    
    # 对 u(r) 归一化：注意归一化要求 ∫|u(r)|²dr = 1
    for i in range(u_eig.shape[1]):
        norm = np.sqrt(np.sum(np.abs(u_eig[:, i])**2) * dr)
        u_eig[:, i] /= norm
    
    # 根据 u(r)=r R(r) 求径向波函数 R(r)=u(r)/r
    R_eig = u_eig / r_in[:, np.newaxis]
    # 顺便对 R(r) 再归一化：归一化要求 ∫|R(r)|² r² dr = 1
    for i in range(R_eig.shape[1]):
        norm = np.sqrt(np.sum(np.abs(R_eig[:, i])**2 * r_in**2) * dr)
        R_eig[:, i] /= norm

    return r_in, energies, u_eig, R_eig

def main():
    # 以 l=0（s 轨道） 为例进行求解
    l = 0
    N = 1000      # 网格点数
    r_max = 30.0  # 半径上限
    r, energies, u_eig, R_eig = solve_hydrogen_atom(l=l, N=N, r_max=r_max)
    
    print("氢原子（l=0）前 3 个能量本征值：")
    # 氢原子解析能量：E_n = -1/(2n²)（原子单位下）
    for i in range(3):
        n = i + 1   # 对于 l=0，第一个本征对应 n=1 (1s), 第二个 n=2 (2s)……
        E_analytic = -1.0 / (2 * n**2)
        print(f"n={n}: E_numerical = {energies[i]:.6f}, E_analytic = {E_analytic:.6f}")
    
    # 绘制 1s 态的径向波函数 R(r) 与解析解对比：
    plt.figure(figsize=(10, 6))
    # 数值得到的 1s 态（第一个本征态）
    plt.plot(r, R_eig[:, 0], 'b-', label="Numerical 1s")
    # 解析 1s 态波函数（在原子单位下，归一化形式为 R_1s(r)=2 exp(-r)）
    R1s_analytic = 2 * np.exp(-r)
    # 为确保归一化一致，对解析解同样归一化：∫|R(r)|²r²dr=1
    dr = r[1]-r[0]
    norm = np.sqrt(np.sum(R1s_analytic**2 * r**2) * dr)
    R1s_analytic /= norm
    plt.plot(r, R1s_analytic, 'r--', label="Analytic 1s")
    plt.xlabel("r (a.u.)")
    plt.ylabel("R(r)")
    plt.title("氢原子 1s 径向波函数比较 (l=0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()
