# %%
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sqpulse import (
    load_models,
    save_models,
    Transmon,
    ReadoutResonator,
    PulseSequence,
    FlatTopPulse,
    GaussianPulse,
    Measurement,
    ns,
    us,
    MHz,
    GHz,
)

# 1. 自动定位并加载芯片与耦合模型
config_path = Path(__file__).parent / "chip_config.json"
if not config_path.is_file():
    config_path = Path("./chip_config.json")

models = load_models(config_path)
print(f"成功加载组件容器: {models}")
print(f"芯片名称: {models.chip_name}")
print(f"包含的量子比特: {list(models.transmons.keys())}")
print(f"包含的谐振腔:   {list(models.resonators.keys())}")
if models.couplings:
    print(f"包含的耦合通道:   {[c.get('name', str(c.get('modes', []))) for c in models.couplings]}\n")

# 获取可调频率量子比特 q1 与读出腔 r1
q1 = models["q1"]
r1 = models["r1"]
coup_q1_r1 = models.get_coupling("q1", "r1")

print(f"=== 组件物理参数 ===")
print(f"  比特 {q1.name}: 最大频率 f_q,max = {q1.f_q/1e9:.2f} GHz, 非谐性 alpha = {q1.alpha/1e6:.1f} MHz, 不对称度 d = {q1.d}, v_phi0 = {q1.v_phi0} V")
print(f"  谐振腔 {r1.name}: 裸频率 f_r = {r1.f_r/1e9:.4f} GHz, 总线宽 kappa = {r1.kappa_hz/1e6:.2f} MHz, 外耦合 kappa_ext = {r1.kappa_ext_hz/1e6:.2f} MHz")
if coup_q1_r1:
    print(f"  耦合通道: {coup_q1_r1.get('name')}, 耦合强度 g = {coup_q1_r1.get('g')}, 标称色散频移 chi = {coup_q1_r1.get('chi')}\n")

# %%
# 2. 编排并绘制中心对齐脉冲序列 (Z-Flux 脉冲包裹 Readout 探测脉冲)
# 实验原理：Z-Flux 脉冲持续时间较长，Readout 脉冲居中对其，
# 从而确保微波读出腔探测期间，比特完全稳定工作在设定的磁通偏置平顶区。
seq = PulseSequence(name="flux_assisted_readout")

# 设定某个目标磁通偏置，例如 0.25 Phi_0，并通过 q1.flux_to_voltage 计算对应所需的 AWG 输出电压
flux_target = 0.25
v_amp = q1.flux_to_voltage(flux_target)

# Z-flux 偏置脉冲（800 ns，含 20 ns 上升/下降沿）
flux_pulse = FlatTopPulse(duration=800 * ns, amp=v_amp, ramp_time=20 * ns)
# Readout 探测脉冲（500 ns，含 20 ns 边缘，短于 Z-flux 脉冲）
ro_pulse = FlatTopPulse(duration=500 * ns, amp=0.5, ramp_time=20 * ns)

# 使用 align_center 实现中心对齐：Readout 脉冲将居中位于 [150 ns, 650 ns]，前后各留有 150 ns 稳态裕量
seq.align_center((q1.z, flux_pulse), (q1.ro, ro_pulse))

print(f"脉冲序列总时长: {seq.duration * 1e9:.1f} ns")
print(f"  Z-flux 偏置脉冲宽度: 800 ns, 设定偏置电压: {v_amp:.3f} V (对应 {flux_target} Phi_0)")
print(f"  Readout 探测脉冲宽度: 500 ns (中心对齐于 Z 偏置有效区内)")

seq.plot()
plt.show()

# %%
# 3. 模拟谐振腔透射谱随 Z Flux 调谐的 2D 响应图 (S21 vs. Z Flux 2D Colormap)
# 物理原理：
# 当改变磁通偏置时，SQUID 比特频率 f_q(Phi) 发生周期性变化；
# 比特频率的调谐改变了失谐量 Delta(Phi) = f_q(Phi) - f_r，进而使腔频产生周期性的色散频移：
# f_r_eff(Phi) = f_r + g^2 / (f_r - f_q(Phi))
# 测量谐振腔 S21 透射谱即可观测到清晰的周期性谐振下陷拱线（Resonator Flux Arc）。

# 扫描通量偏置范围：-0.6 到 +0.6 Phi_0 (涵盖完整周期及极值点)
flux_sweep = np.linspace(-2.6, 2.6, 121)
# 扫描微波探针频率范围：裸腔 7.05 GHz 附近
freq_sweep = np.linspace(7.04 * GHz, 7.06 * GHz, 151)

# 获取传输矩阵 (Shape: len(freqs) x len(fluxes))
s21_matrix = r1.s21_vs_flux(freqs=freq_sweep, transmon=q1, fluxes=flux_sweep)
mag_db = 20.0 * np.log10(np.abs(s21_matrix) + 1e-15)
phase_deg = np.degrees(np.angle(s21_matrix))

# 绘制 2D 颜色图
fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(13, 5))
extent = [flux_sweep[0], flux_sweep[-1], freq_sweep[0] / 1e9, freq_sweep[-1] / 1e9]

im1 = ax_mag.imshow(
    mag_db,
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
ax_mag.set_xlabel(r"Z-Flux $\Phi / \Phi_0$", fontsize=11)
ax_mag.set_ylabel("Probe Frequency (GHz)", fontsize=11)
ax_mag.set_title(r"Transmission Magnitude $|S_{21}|$ (dB)", fontsize=12)
cb1 = fig.colorbar(im1, ax=ax_mag)
cb1.set_label("dB")

im2 = ax_phase.imshow(
    phase_deg,
    extent=extent,
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
)
ax_phase.set_xlabel(r"Z-Flux $\Phi / \Phi_0$", fontsize=11)
ax_phase.set_ylabel("Probe Frequency (GHz)", fontsize=11)
ax_phase.set_title(r"Transmission Phase $\angle S_{21}$ (deg)", fontsize=12)
cb2 = fig.colorbar(im2, ax=ax_phase)
cb2.set_label("deg")

fig.suptitle(f"Resonator Flux Arc: {r1.name} coupled to tunable {q1.name}", fontsize=13, y=0.98)
fig.tight_layout()
plt.show()

# %%
# 4. 提取典型通量工作点（甜点 Phi=0 与半磁通点 Phi=0.5）的 1D 透射截面切片
idx_sweet = np.argmin(np.abs(flux_sweep - 0.0))
idx_half = np.argmin(np.abs(flux_sweep - 0.5))

plt.figure(figsize=(8, 4.5))
plt.plot(freq_sweep / 1e9, mag_db[:, idx_sweet], label=f"Sweet Spot (Φ = 0.0, f_q = {q1.frequency_at_flux(0.0)/1e9:.2f} GHz)", color="#1f77b4", lw=2)
plt.plot(freq_sweep / 1e9, mag_db[:, idx_half], label=f"Half Flux (Φ = 0.5 Φ₀, f_q = {q1.frequency_at_flux(0.5)/1e9:.2f} GHz)", color="#d62728", lw=2, linestyle="--")
plt.xlabel("Probe Frequency (GHz)", fontsize=11)
plt.ylabel("|S₂₁| (dB)", fontsize=11)
plt.title("Resonator 1D Spectrum at Different Flux Biases (Dip Mode)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# %%
# 5. 全时序微波脉冲动力学仿真 (Time-Domain Pulse Dynamics via Measurement.run)
# 原理说明：
# 前述分析采用稳态解析公式，此处调用 SQPulse 统一仿真接口 Measurement.run(backend="dispersive")，
# 真实求解含时海森堡-朗之万微分方程 (Langevin Equation)：
# dα(t)/dt = - [i·2π(f_ro - f_r(t)) + κ/2]·α(t) - i·ε_ro(t)
# 其中 f_r(t) 由 q1.z 通道的时域波形动态计算，真实呈现腔内光子充场 (Ring-up) 与衰减 (Ring-down)。

# 计算 Phi = 0.25 Phi_0 处的理论谐振腔频率
f_res_biased = r1.effective_frequency_at_flux(q1, flux=0.25, g=r1.g_hz, qubit_state=0)

disp_result = Measurement.run(
    target=q1,
    sequence=seq,
    resonator=r1,
    backend="dispersive",
    f_ro=f_res_biased,
    g=r1.g_hz,
    qubit_state=0,
)

print(f"\n=== 全时序脉冲动力学仿真 (Measurement.run) ===")
print(f"  探测载频: {f_res_biased / 1e9:.6f} GHz (对应 Phi = 0.25 Phi_0)")
print(f"  提取出的动态透过率 S21: {disp_result.s21:.4f}")
print(f"  动态透射模长 |S21|: {abs(disp_result.s21):.4f} ({disp_result.s21_db:.2f} dB), 相位: {disp_result.s21_phase_deg:.1f}°")

# 5.1 绘制时域微波信号与腔内光子数动态演化
fig, (ax_wave, ax_photons) = plt.subplots(1, 2, figsize=(13, 4.5))
disp_result.plot_output_waveforms(state=0, ax=ax_wave)
disp_result.plot_trajectories(ax=ax_photons)
fig.suptitle(f"Dynamic Pulse Evolution: {r1.name} under Concurrent Z-Flux and Readout Drives", fontsize=12)
fig.tight_layout()
plt.show()

# 5.2 物理时序对比：为什么 Z-Flux 脉冲必须比 RO 脉冲更长？
# 对比方案 A（推荐）：800 ns 长通量脉冲，平顶平稳覆盖 500 ns 读取区间；
# 对比方案 B（过窄）：400 ns 短通量脉冲，在读取脉冲进行到一半时通量提前撤出，腔频发生剧烈突变（Chirp）。
short_flux_pulse = FlatTopPulse(duration=400 * ns, ramp_time=20 * ns, amp=v_amp)
seq_short = PulseSequence().align_center((q1.z, short_flux_pulse), (q1.ro, ro_pulse))

disp_short = Measurement.run(
    target=q1,
    sequence=seq_short,
    resonator=r1,
    backend="dispersive",
    f_ro=f_res_biased,
    g=r1.g_hz,
    qubit_state=0,
)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
t_ns = disp_result.times * 1e9
ax.plot(t_ns, np.abs(disp_result.output_fields[0]), label=f"Properly Biased (800 ns Flux: stable, S21={disp_result.s21_db:.2f} dB)", color="#1f77b4", lw=2)
ax.plot(disp_short.times * 1e9, np.abs(disp_short.output_fields[0]), label=f"Under-Biased (400 ns Flux: jumps mid-readout, S21={disp_short.s21_db:.2f} dB)", color="#d62728", lw=2, linestyle="--")
ax.set_xlabel("Time (ns)", fontsize=11)
ax.set_ylabel(r"Transmitted Field Amplitude $|V_{\mathrm{out}}(t)|$", fontsize=11)
ax.set_title("Timing Physics: Why Z-Flux Pulse Must Cover Readout Pulse", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
plt.show()


# %%
