# 架构与输入输出契约

## 阶段

| 阶段 | 模块 | 输入 | 输出 |
| --- | --- | --- | --- |
| priors | `priors/png_masks.py` | 逐视图 RGBA 实例 PNG 或 npz | `{view: {inst_id: bool(H,W)}}` |
| colmap | `geometry/colmap_io.py` | COLMAP text 稀疏模型 | cameras / images / points3D |
| match ★ | `matching/projection_hungarian.py` | 掩码 + 位姿 + 稀疏点 | 全局叶片实例 + diagnostics |
| propagate | `geometry/propagate.py` | 全局实例 + 稠密点云 | 逐点器官标签 |
| postprocess | `organs/postprocess.py` | 标签 + 点云 | 合并后标签 + 测量/可视化掩码 |
| traits | `traits/measure.py` | 标签 + 点云（cm） | traits.csv / plant_traits.json |
| quality | `quality/reports.py` | 上述全部 | matching/postprocess diagnostics |

## 匹配核心（projection_hungarian）的机制

1. **共享投影支撑**：稀疏点若能投影进某视图某实例掩码，则属于该实例的支撑集。
2. **逐视角对打分**：支撑集 Jaccard。
3. **一对一指派**：每对视角独立做匈牙利匹配，天然防止"一个实例吞对面多个"。
4. **同视图冲突保护**：按分数降序用并查集建全局组件；若一条边会使同一组件在同一视图
   占两个实例，拒绝该边并记录理由。
5. **低支持 warning**：支撑点数 < `min_support_points` 的全局实例只告警不合并。

### 为什么这样设计能防"身份串联"

两片接触/重叠叶子的掩码在不同视角可能互相吸引。一对一指派限制每对视角的贪心吞并；
同视图冲突保护在全局层面兜底——任何会导致"一个物理叶片在同一视角出现两个身份"
的合并都被显式拒绝，且拒绝理由进 diagnostics 可复查。

## 数据集配置契约

`configs/datasets/<name>.toml`：

```toml
[dataset]
name = "my_plant"
sparse_dir = "path/to/sparse/0"     # COLMAP text 格式三件套
masks_dir = "path/to/masks"         # 每视图一个 npz：键为实例 id，值 bool(H,W)
dense_path = "path/to/dense.npy"    # (N,4)：xyz + 可选真值标签
stem_label = 0                      # 茎的标签（5plant 约定：茎实例统一 0）
```

## 坐标与单位

- 假定输入点云坐标单位为**厘米**（与 5plant 数据集一致）。
- COLMAP 四元数约定 (w, x, y, z)，外参 x_cam = R·x_world + t。
