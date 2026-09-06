# SQPulse: 超导量子脉冲工程与动力学仿真库

**SQPulse** 是一个专为超导量子计算（Transmon Qubit）、微波脉冲工程与含时量子动力学模拟而设计的现代 Python 库，提供了直观、模块化和显式调用的 API。

---

## 核心特性

1. **丰富的脉冲波形与时频域双重视角**：
   - 支持高斯脉冲 (`GaussianPulse`)、升余弦脉冲 (`CosinePulse` / Hann 窗)、柯西-洛伦兹脉冲 (`LorentzianPulse`)、理想方波 (`SquarePulse`)、平顶平滑脉冲 (`FlatTopPulse`)、双曲正割脉冲 (`SechPulse`)、DRAG 导数修正脉冲 (`DRAGPulse`) 及任意自定义波形 (`CustomPulse`)。
   - 内置一键式时域与频域双联可视化 (`pulse.plot(domain="both")`)，轻松分析脉冲频谱、半高全宽（FWHM）、谱泄露（Spectral Leakage）与旁瓣衰减。

2. **严谨的受驱动 Transmon 物理模型**：
   - 支持多能级截断 ($d \ge 2$)，准确刻画更高能级态 $|2\rangle$ 的弱跃迁与泄露。
   - 包含量子频率 $f_q$、非谐性 $\alpha$、微波驱动正交算符 $H_x, H_y$。
   - 内置包含能量弛豫 $T_1$、纯退相位 $T_\phi$（根据 $T_2$ 换算）与热激发 $n_{\text{th}}$ 的完整 Lindblad 耗散主方程。

3. **显式的时间轴脉冲编排调度器 (`PulseSequence`)**：
   - 无隐式全局副作用，通过链式或方法调用清晰添加脉冲 (`add`)、插入延时 (`delay`) 与多通道时钟同步 (`sync`)。
   - 一键绘制多通道时序图 (`seq.plot()`)，并可直接编译为 QuTiP 5 的 `QobjEvo` 含时哈密顿量。

4. **动力学主方程演化与量子态分析 (`Simulator`)**：
   - 基于现代 QuTiP 5 求解器快速求解密度矩阵或纯态演化。
   - 提供各能级粒子数布居随时间演化曲线 (`plot_populations`)、Bloch 坐标曲线 (`plot_bloch_vector`) 及 3D Bloch 球轨迹投影 (`plot_bloch_sphere`)。

5. **标准化经典量子实验协议 (`experiments`)**：
   - **Amplitude / Time Rabi**：自动扫描驱动幅度，通过正弦拟合标定 $\pi$ 脉冲与 $\pi/2$ 脉冲。
   - **$T_1$ 弛豫测量**：施加 $\pi$ 脉冲后扫描延时，通过指数衰减拟合提取 $T_1$ 寿命。
   - **Ramsey 干涉测量**：$\pi/2 - \tau - \pi/2$ 干涉序列，通过阻尼正弦拟合提取 $T_2^*$ 及微波失谐量 $\Delta$。

---

## 快速上手

### 1. 查看脉冲时域与频域特征

```python
import matplotlib.pyplot as plt
from sqpulse import GaussianPulse, CosinePulse, SquarePulse, compare_pulses

# 定义一个 40 ns 的高斯脉冲
p_gauss = GaussianPulse(duration=40.0, amp=1.0, chop=4.0)

# 一键展示时域波形与频域 FFT 功率谱
p_gauss.plot(domain="both")
plt.show()

# 对比多种波形的抗高频谱泄露性能
p_cos = CosinePulse(duration=40.0, amp=1.0)
p_sq = SquarePulse(duration=40.0, amp=1.0)
compare_pulses([p_gauss, p_cos, p_sq], domain="both")
plt.show()
```

### 2. 定义受驱动 Transmon 并编排脉冲序列

```python
import matplotlib.pyplot as plt
from sqpulse import Transmon, PulseSequence, GaussianPulse, Simulator

# 1. 定义 Transmon (5.0 GHz, 非谐性 -250 MHz, 3能级, T1=25us, T2=18us)
q = Transmon("q0", f_q=5.0, alpha=-0.25, levels=3, t1=25000.0, t2=18000.0)

# 2. 编排脉冲序列
seq = PulseSequence(name="xy_drive")
seq.add(q.drive, GaussianPulse(duration=30.0, amp=0.08))
seq.delay(q.drive, 20.0)
seq.add(q.drive, GaussianPulse(duration=30.0, amp=0.08, phase=1.5708)) # 绕 Y 轴驱动

# 3. 绘制时序图
seq.plot()
plt.show()

# 4. 动力学演化
result = Simulator.run(q, seq)
result.plot_populations()
result.plot_bloch_vector()
plt.show()
```

### 3. 运行量子实验测量

```python
import matplotlib.pyplot as plt
from sqpulse import Transmon, GaussianPulse, RabiExperiment, T1Experiment, RamseyExperiment

q = Transmon("q0", f_q=5.0, alpha=-0.25, t1=20000.0, t2=15000.0)

# 3.1 振幅 Rabi 标定 pi 脉冲幅度
rabi_res = RabiExperiment.amplitude_rabi(q, pulse_type=GaussianPulse, duration=40.0)
print(f"标定得到的 pi 脉冲幅度: {rabi_res.amp_pi:.5f}")
rabi_res.plot()
plt.show()

# 3.2 T1 弛豫测量
pi_pulse = GaussianPulse(duration=40.0, amp=rabi_res.amp_pi)
t1_res = T1Experiment.run(q, pi_pulse=pi_pulse)
print(f"拟合得到的 T1 寿命: {t1_res.t1_fit:.1f} ns")
t1_res.plot()
plt.show()

# 3.3 Ramsey 干涉实验
pi2_pulse = GaussianPulse(duration=40.0, amp=rabi_res.amp_pi_half)
ramsey_res = RamseyExperiment.run(q, pi_half_pulse=pi2_pulse, detuning=0.002) # 2 MHz 失谐
print(f"拟合测得 T2*: {ramsey_res.t2_star:.1f} ns, 失谐: {ramsey_res.fitted_detuning*1e3:.2f} MHz")
ramsey_res.plot()
plt.show()
```

---

## 运行测试

使用 `pytest` 运行单元测试套件：

```bash
uv run pytest
```
