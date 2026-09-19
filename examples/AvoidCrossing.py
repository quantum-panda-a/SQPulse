# %%
import numpy as np
import matplotlib.pyplot as plt

from sqpulse import (
    Transmon,
    QuantumSystem,
    Measurement,
    PulseSequence,
    SquarePulse,
    FlatTopPulse,
    FluxISWAP,
    GHz,
    MHz,
    ns,
)

# 1. 定义两个物理比特
# q0: SQUID 可调频率比特（Sweet Spot 处 5.0 GHz）
q0 = Transmon(
    name="q0",
    f_q=5.0 * GHz,
    alpha=-250.0 * MHz,
    d=0.3,              # SQUID 不对称参数 (可调)
    flux_offset=0.0,
    t1=30.0e-6,
    t2=20.0e-6,
    levels=3,
)

# q1: 单结固定频率比特（4.6 GHz）
q1 = Transmon(
    name="q1",
    f_q=4.6 * GHz,
    alpha=-240.0 * MHz,
    d=1.0,              # 单结 (固定频率)
    t1=30.0e-6,
    t2=20.0e-6,
    levels=3,
)

# 2. 构建复合系统并加入电容横向交换耦合 g = 25 MHz
sys = QuantumSystem([q0, q1], name="coupled_pair")
g_coupling = 25.0 * MHz
sys.add_capacitive_coupling(q0, q1, g=g_coupling)

# 计算理论共振磁通偏置 (f_q0(Phi_res) = f_q1 = 4.6 GHz)
phi_res = FluxISWAP.calculate_resonance_flux(q0, q1.f_q)
print(f"理论共振点磁通偏置: Phi_res = {phi_res:.4f} Phi_0")

# %%
# 3. 设置 2D 扫描网格 (磁通 vs 微波驱动频率)
fluxes = np.linspace(0.165, 0.205, 21)               # 覆盖共振点两侧的磁通范围
freqs = np.linspace(4.54 * GHz, 4.66 * GHz, 31)      # 探针微波扫描频率范围
p_exc_grid = np.zeros((len(fluxes), len(freqs)))

probe_duration = 300 * ns
probe_amp = 0.02  # 弱微波探针幅度

print("开始 2D 频谱扫描...")
for i, phi in enumerate(fluxes):
    for j, f_d in enumerate(freqs):
        # 4. 构造脉冲时序：
        # - 在 q0.z 施加平顶磁通脉冲调节其能级
        # - 在 q0.xy 施加微波连续弱探针
        seq = PulseSequence("spec_probe")
        seq.add(q0.z, FlatTopPulse(duration=probe_duration + 20 * ns, amp=phi, ramp_time=5 * ns))
        seq.add(q0.xy, SquarePulse(duration=probe_duration, amp=probe_amp))

        # 5. 调用 Measurement.run，target 传入复合系统 sys
        # f_d 作为微波旋转坐标系参考频率（模拟外加探针载波频率）
        res = Measurement.run(target=sys, sequence=seq, f_d=float(f_d), dt=2e-9)

        # 提取系统激发概率 (P_exc = 1 - P_|00>)
        p_exc_grid[i, j] = 1.0 - res.final_population("00")

# 6. 绘制 2D 免交叉能谱图
plt.figure(figsize=(9, 5.5))
plt.pcolormesh(
    freqs / GHz,
    fluxes,
    p_exc_grid,
    shading="auto",
    cmap="viridis",
)
cbar = plt.colorbar()
cbar.set_label("Total Excitation Probability $P_{\\text{exc}}$", fontsize=11)

# 标出未耦合时的理论曲线作为对照
f_q0_bare = [q0.frequency_at_flux(phi) / GHz for phi in fluxes]
plt.plot(f_q0_bare, fluxes, "w--", alpha=0.5, label="Bare $q_0(\\Phi)$")
plt.axvline(q1.f_q / GHz, color="r", ls="--", alpha=0.5, label="Bare $q_1$ (Fixed)")

plt.xlabel("Probe Drive Frequency (GHz)", fontsize=11)
plt.ylabel("External Flux Bias $\\Phi$ ($\\Phi_0$)", fontsize=11)
plt.title(f"Two-Qubit Avoided Crossing Spectroscopy ($g = {g_coupling / MHz:.0f}$ MHz)", fontsize=12)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
# %%
