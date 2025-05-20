import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 使matplotlib支持中文显示
plt.rcParams["font.family"] = "SimHei"  
# 防止负号显示异常
plt.rcParams["axes.unicode_minus"] = False

def solve_infinite_well(N=1000, L=1.0):
    """
    对一维无限深势阱问题求解数值本征值和本征态。
    参数：
      N : 网格点数（包括边界点）
      L : 势阱长度
    返回：
      x_in        : 内部网格点，去除两端边界
      energies    : 数值求解的能量本征值（升序排列）
      wavefuncs   : 对应的本征态（每列对应一个归一化的波函数）
    """
    # 将区间 [0, L] 均匀离散为 N 个点
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    # 对于 Dirichlet 边界条件，将端点剔除，只对内部点构造矩阵
    x_in = x[1:-1]
    N_in = len(x_in)
    
    # 使用有限差分法构造二阶导数的离散表示：
    # 中心差分格式： psi''(x_i) ≈ (psi[i+1] - 2 psi[i] + psi[i-1]) / dx^2.
    # 对应动能算符： -1/2 * d^2/dx^2 ——>
    # 对角元： 1/dx^2, 非对角元： -1/(2*dx^2)
    diag = np.full(N_in, 1.0 / dx**2)
    off_diag = np.full(N_in - 1, -1.0 / (2 * dx**2))
    H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    # 无限深势阱内部势能为 0，只需要构造动能项即可
    energies, wavefuncs = eigh(H)
    
    # 离散归一化： ∫|ψ(x)|² dx ≈ sum(|ψ|²)*dx
    for i in range(wavefuncs.shape[1]):
        norm = np.sqrt(np.sum(np.abs(wavefuncs[:, i])**2) * dx)
        wavefuncs[:, i] /= norm
        
    return x_in, energies, wavefuncs

def solve_harmonic_oscillator(N=1000, x_max=8.0, omega=1.0):
    """
    对一维谐振子问题求解数值本征值和本征态。
    参数：
      N      : 网格点数（包括边界点）
      x_max  : 区间的半宽度，求解域为 [-x_max, x_max]
      omega  : 谐振子角频率（此处取 omega=1）
    返回：
      x_in        : 内部网格点
      energies    : 数值求解的能量本征值（升序排列）
      wavefuncs   : 对应的本征态（每列对应一个归一化的波函数）
    """
    # 将区间 [-x_max, x_max] 均匀离散为 N 个点
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]
    # 剔除边界，从内部构造求解矩阵
    x_in = x[1:-1]
    N_in = len(x_in)
    
    # 构造动能项同上
    diag = np.full(N_in, 1.0 / dx**2)
    off_diag = np.full(N_in - 1, -1.0 / (2 * dx**2))
    T = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    # 构造谐振子势能： V(x) = 0.5 * omega^2 * x^2
    V = 0.5 * omega**2 * x_in**2
    H = T + np.diag(V)
    
    energies, wavefuncs = eigh(H)
    
    # 离散归一化
    for i in range(wavefuncs.shape[1]):
        norm = np.sqrt(np.sum(np.abs(wavefuncs[:, i])**2) * dx)
        wavefuncs[:, i] /= norm

    return x_in, energies, wavefuncs

def main():
    # ---------------------------
    # 无限深势阱的求解与对比
    # ---------------------------
    L = 1.0      # 势阱长度
    x_well, energies_well, wf_well = solve_infinite_well(N=1000, L=L)
    print("【无限深势阱】前5个能量本征值：")
    for n in range(5):
        # 理论上： E_n = n^2 * pi^2 / 2, n=1,2,...
        E_exact = ((n + 1)**2 * np.pi**2) / 2
        print(f"n={n+1}: E_num = {energies_well[n]:.6f}, E_exact = {E_exact:.6f}")
    
    # 绘制前三个本征态
    plt.figure(figsize=(10, 5))
    for n in range(3):
        plt.plot(x_well, wf_well[:, n], label=f'n={n+1}')
    plt.title("无限深势阱波函数")
    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # ---------------------------
    # 谐振子的求解与对比
    # ---------------------------
    x_max = 8.0  # 取足够宽的区域使得边界效应可以忽略
    omega = 1.0  # 谐振子角频率
    x_ho, energies_ho, wf_ho = solve_harmonic_oscillator(N=1000, x_max=x_max, omega=omega)
    print("\n【谐振子】前5个能量本征值：")
    for n in range(5):
        # 理论上： E_n = (n + 1/2), n=0,1,...
        E_exact = n + 0.5
        print(f"n={n}: E_num = {energies_ho[n]:.6f}, E_exact = {E_exact:.6f}")
    
    # 绘制前三个本征波函数
    plt.figure(figsize=(10, 5))
    for n in range(3):
        plt.plot(x_ho, wf_ho[:, n], label=f'n={n}')
    plt.title("谐振子波函数")
    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()
