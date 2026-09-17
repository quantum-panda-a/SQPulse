import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sqpulse import (
    load_models,
    save_models,
    Transmon,
    PulseSequence,
    GaussianPulse,
    FlatTopPulse,
    Measurement,
    ns,
    us,
    MHz,
    GHz,
)


def main():
    config_path = Path(__file__).parent / "chip_config.json"
    print(f"=== 1. 加载芯片模型配置文件: {config_path.name} ===")

    # 1. 一行代码加载所有模型
    models = load_models(config_path)
    print(f"成功加载组件容器: {models}")
    print(f"包含的量子比特: {list(models.transmons.keys())}")
    print(f"包含的谐振腔:   {list(models.resonators.keys())}\n")

    # 2. 检查加载后对象的物理参数（已自动由字符串单位转为 SI 单位）
    q0: Transmon = models["q0"]
    r0 = models["r0"]

    print("=== 2. 查看 q0 物理参数 ===")
    print(f"  名称: {q0.name}")
    print(f"  跃迁频率 f_q:       {q0.f_q / 1e9:.3f} GHz (SI 内部值: {q0.f_q:.1f} Hz)")
    print(f"  非谐性 alpha:       {q0.alpha / 1e6:.1f} MHz")
    print(f"  结不对称度 d:       {q0.d} ({'可调 SQUID' if q0.d < 1.0 else '固定频率单结'})")
    print(f"  弛豫时间 T1:        {q0.t1 * 1e6:.1f} us")
    print(f"  去相时间 T2:        {q0.t2 * 1e6:.1f} us")
    print(f"  等效热浴温度 T:     {q0.temperature * 1e3:.1f} mK")
    print(f"  热激发占有率 n_th:  {q0.thermal_population:.4%} (自动由 Bose-Einstein 分布计算)")
    print(f"  磁通周期电压 v_phi0: {q0.v_phi0:.2f} V/Phi_0\n")

    print("=== 3. 查看 r0 读出谐振腔参数 ===")
    print(f"  名称: {r0.name}")
    print(f"  腔频率 f_r:         {r0.f_r / 1e9:.3f} GHz")
    print(f"  总线宽 kappa/2pi:   {r0.kappa_hz / 1e6:.2f} MHz")
    print(f"  色散频移 chi/2pi:   {r0.chi_hz / 1e6:.2f} MHz\n")

    # 3. 使用从 JSON 加载的模型进行脉冲编排与动力学演化
    print("=== 4. 运行含时动力学演化仿真 ===")
    seq = PulseSequence(name="xy_drive_and_z_bias")

    # 施加 30 ns 高斯脉冲
    seq.add(q0.xy, GaussianPulse(duration=30 * ns, amp=0.5))
    seq.sync()

    # 施加 50 ns Z 偏置磁通脉冲 (电压驱动自动折算为超导环净磁通)
    seq.add(q0.z, FlatTopPulse(duration=50 * ns, amp=0.1 * q0.v_phi0, ramp_time=5 * ns))

    sim_res = Measurement.run(q0, seq, dt=1.0 * ns)
    p0 = sim_res.final_population(0)
    p1 = sim_res.final_population(1)
    p2 = sim_res.final_population(2)

    print(f"脉冲序列总时长: {seq.duration * 1e9:.1f} ns")
    print(f"演化后能级布居: |0> = {p0:.4f}, |1> = {p1:.4f}, |2> = {p2:.4f}\n")

    # 4. 演示标定参数更新与重新保存
    print("=== 5. 标定后参数导出与回写 ===")
    # 模拟经过实验标定后测得更新的 T1 值
    q0.t1 = 32.5e-6

    # 导出可读字典
    exported_dict = q0.to_dict(human_readable=True)
    print(f"q0 导出为人类可读配置字典:\n  {exported_dict}\n")

    # 保存整颗芯片配置到输出文件
    output_file = Path(__file__).parent / "calibrated_chip_output.json"
    save_models(models, output_file, human_readable=True)
    print(f"已将更新后的芯片模型保存至: {output_file.name}")

    # 验证重读
    reloaded = load_models(output_file)
    print(f"重读验证 q0 新 T1: {reloaded['q0'].t1 * 1e6:.1f} us")

    # 清理演示临时文件
    output_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
