# Organ3DNet 复现指南

本机完整复现流程（2026-11 验证通过）。

## 环境

- WSL2 Ubuntu-24.04，conda 环境：python 3.10 / torch 1.12.1+cu113 / MinkowskiEngine（源码编译）/ pointnet2 / pytorch-lightning 1.7.2
- GPU：RTX 5070 Ti（16 GB）；官方推荐 ≥24 GB，显存越大越好
- 注意：必须 torch 1.12 + cu113 这套老组合，MinkowskiEngine 按官方 README 的 commit 02fc608 编译

## 数据与权重

1. 数据集：zenodo DOI 10.5281/zenodo.17165870（5plant_xyz_ins_sem.rar，889 株）
2. 预训练权重：官方 checkpoint（last-epoch.ckpt）
3. 预处理四步（见 Organ3DNet README）：
   - 坐标归一化到 [0,30]
   - 训练集 10 倍增广（整株 + FPS 9 中心立方体裁剪）
   - 边缘点标注（3DEPS 采样用）
   - 转 (N,7) npy

## 评测命令

```bash
python main_instance_segmentation.py \
  general.train_mode=false general.export=true \
  general.checkpoint=/path/to/last-epoch.ckpt \
  data/datasets=5plant general.area=2 data.voxel_size=0.15 \
  general.num_targets=11 data.num_labels=10
python offline_eval.py   # 官方 benchmark 评分
```

## 已知坑

- 显存不足时推理会静默变慢或 OOM，README 明说性能对显存敏感
- benchmark 脚本用了 np.int/np.float 等已删除别名，需打兼容补丁（见我们 offline_eval.py 开头）
- shell 变量传 hydra 参数时注意引号，空 data_dir 会被静默解析为 "." 并提前退出
