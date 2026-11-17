# plant-organ-phenotyping

多视角 RGB 驱动的器官级植物三维表型管线（独立复现实现）。
Independent re-implementation of an auditable organ-level plant 3D phenotyping pipeline:
cross-view leaf identity matching → dense label propagation → organ traits → quality reports.

## 这条管线做什么

```
frozen 2D priors(逐视图叶片掩码)
  → COLMAP 位姿 + 稀疏点
  → projection_hungarian 跨视图身份匹配   ★ 核心：共享投影支撑 + 匈牙利一对一 + 同视图冲突保护
  → 稠密点云身份传播(投影投票)
  → 器官后处理(碎片合并 / 茎基部保护 / 测量-可视化点云分离)
  → 表型量测(叶面积/叶长/叶宽/叶倾角/株高/茎径)
  → 质量报告(matching_diagnostics.json 等，全程可审计)
```

## 快速开始

```bash
pip install numpy scipy pillow
python examples/make_synthetic_dataset.py     # 生成 3 视角 6 叶的合成演示数据
python tests/test_matching.py                 # 单元测试：身份恢复 + 冲突保护
python scripts/pipeline.py --dataset synthetic_demo   # 端到端六阶段
```

产物在 `outputs/synthetic_demo/`：`traits.csv`、`plant_traits.json`、
`matching_diagnostics.json`、`postprocess_diagnostics.json`。

**有真实视频？** 直接出点云和单株分析：

```bash
python tools/scan_video.py --video plant.mp4 --out recon1   # 视频→稠密点云
python tools/analyze_plant.py recon1/dense/fused.ply recon1/main --radius 1.2
```

拍摄规范与实测数据见 [docs/VIDEO.md](docs/VIDEO.md)。

**做方法研究？** 5plant 真值基准 + 参数扫描：

```bash
python examples/make_5plant_dataset.py <5plant_labeled.txt> my_plant 12
python examples/sweep_match_params.py my_plant   # 匹配机制 A/B 对比
```

## 设计原则（与 GeoProMatch 论文契约一致）

1. **可审计**：每个全局器官可追溯到具体视图的实例支撑；被拒绝的匹配边记录理由；
   低支持实例给 warning，不静默合并。
2. **测量与可视化分离**：`measurement point cloud` 与 `visualization point cloud` 分开交付。
3. **配置即信任边界**：阶段命令从 `configs/` 的 TOML 展开，argv 方式执行，不走 shell。

## 本仓库与相关项目的关系

- **Organ3DNet**（Huang2002200/Organ3DNet）：点云器官分割引擎，本仓库不重复包含；
  复现评测见 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)。
- **GeoProMatch**：竞赛/投稿项目，本文档所述管线为其跨视图身份匹配段的独立复现，
  用于方法验证与教学，与原作者实现无代码共享。

## 目录

```
configs/          管线与数据集配置（TOML）
scripts/pipeline.py  阶段式入口
tools/            视频扫描工具链（视频→点云→单株分析，见 docs/VIDEO.md）
src/organ_phenotyping/
  priors/         2D 先验读取（RGBA 透明 PNG / npz 掩码）
  matching/       projection_hungarian 跨视图匹配 ★
  geometry/       COLMAP 读写、稠密身份传播
  organs/         器官后处理
  traits/         表型量测
  quality/        审计报告输出
examples/         合成 / 5plant 真值基准数据生成器 + 参数扫描
tests/            匹配核心单元测试
docs/             架构、评测、复现、视频扫描文档
```

## 许可

MIT（见 LICENSE）。合成数据与文档同许可。
