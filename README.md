# 三维成矿预测代码

本仓库只包含复现实验所需的程序。钻孔数据库、化验、岩性样本、剖面、地表高程，以及由这些数据算出的概率场和距离场，均未包含。

## 程序

对外入口：

- `lithology_probability.py`：由岩性样本和地表高程生成岩性概率场
- `distance_field.py`：由概率场、解析剖面和剖面参数生成距离场
- `4_矿化预测特征表.py`：给体素贴钻孔标签并写出特征表
- `train_three_models.py`：按钻孔做五折，训练随机森林、XGBoost 和三维 PatchCNN。断裂、矿化蚀变和构造蚀变先按 \(F=\exp(-d/\lambda)\) 转为影响强度，距离大于各自的影响距离后强度为 0；岩性接触为 \(\exp(-d/50)\)

`borehole_split.py`、`ml_mineral_train.py`、`dem_attach.py`、`profile_real_coords.py` 是上面几个入口调用的模块。

## 运行

在本目录下，数据按 `data/README.md` 放好后：

```text
python -u lithology_probability.py
python -u distance_field.py
python -u 4_矿化预测特征表.py
python -u train_three_models.py
```

概率场和距离场的结果写到 `results/`。训练脚本写出 `results/fold_assignments.csv`、`results/fold_metrics.csv`、`results/summary.csv` 和 `results/run.log`。

`4_矿化预测特征表.py` 仍按原工程目录寻找距离场、剖面参数、钻孔库和解析表。训练脚本读取本目录 `data/` 中的特征表、孔号对照、特征缓存和 `cnn_volume.dat`。

## 环境

依赖见 `requirements.txt`。卷积神经网络需要与代码匹配的 PyTorch。
