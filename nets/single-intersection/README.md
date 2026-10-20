# nets/single-intersection —— SUMO 路网与需求

本目录用于放置**自定义路网**（4 相位含保护式左转、每方向 3 车道）。
当前尚未创建，`src/env/sumo_env.py` 会自动回退到 `sumo-rl` 包内置的单路口路网，
因此**框架可以先跑通，不受此目录为空的影响**。

## 规划中的文件

| 文件 | 说明 |
| --- | --- |
| `single-intersection.nod.xml` | 节点定义（十字路口 4 个进口 + 1 个中心节点） |
| `single-intersection.edg.xml` | 边定义（每方向 3 条进口车道：左转 / 直行 / 右转） |
| `single-intersection.net.xml` | 由 `netconvert` 生成的路网（**运行时使用**） |
| `single-intersection.rou.xml` | 车辆需求：各方向泊松到达 + 转向比例 |
| `tls.add.xml` | 信号程序：4 个绿灯相位 + 黄灯过渡 |
| `single-intersection.sumocfg` | 仿真配置（引用上述文件） |

## 生成步骤（实现阶段执行）

```bash
# 1) 由 nod/edg 生成路网（含信号灯）
netconvert --node-files single-intersection.nod.xml \
           --edge-files single-intersection.edg.xml \
           --output-file single-intersection.net.xml

# 2) 用 netedit 可视化检查车道与相位
netedit single-intersection.net.xml

# 3) 运行查看效果
sumo-gui -c single-intersection.sumocfg
```

## 与 SUMO-RL 的对接

- `sumo_env.py` 会优先检测本目录下是否存在
  `single-intersection.net.xml` 与 `single-intersection.rou.xml`；
- 两者都存在 → 使用本地路网（我们的 4 相位方案）；
- 否则 → 回退到 `sumo_rl/nets/single-intersection/`（2 相位，仅用于打通流程）。

> 注意：`sumo-rl` 从 `.net.xml` 中解析**绿灯相位个数**并据此确定动作空间，
> 因此本地路网必须定义好 4 个绿灯相位，动作空间才会是 4。

## 参考

- SUMO 路网生成：<https://sumo.dlr.de/docs/netconvert.html>
- 信号程序 `<tlLogic>`：<https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html>
