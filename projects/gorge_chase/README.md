# 初赛：峡谷追逐 PPO

此目录来自 `代码.zip` 内“初赛代码”的 `flash优化` 副本中的 `code/`，作为现存初赛 PPO 方案保留。尚无足够记录确认它在整个初赛中的精确迭代编号；不据文件夹名称推断成绩或时间顺序。

主要实现：

- `agent_ppo/feature/preprocessor.py`：数值与空间特征、探索和访问记忆、阶段奖励、躲避追击、宝箱/加速/闪现相关处理。
- `agent_ppo/model/model.py`：数值 MLP 与四通道 21×21 局部空间 CNN 融合，输出 16 个动作 logits 和一个价值估计。
- `agent_ppo/algorithm/algorithm.py`：PPO 更新，以及学习率、熵和裁剪相关调度。
- `agent_ppo/workflow/train_workflow.py`：环境交互、样本收集、奖励和指标上报。
- `agent_diy/`：同时保留的扩展模板；独立的 DIY 规则实现见仓库 archives 目录。

在匹配的开悟环境内，以本目录为代码根目录运行：

```bash
python train_test.py
```

脚本默认 `algorithm_name = "ppo"`，用于短训练测试。正式训练使用平台流程及 `conf/configure_app.toml`。项目元数据记录 `gorge_chase`、`15.0.1-comp-normal-lite.26comp`；这是来源文件中的版本标识，不保证当前平台仍提供相同环境。

源码和配置内容按原包保留，未附带模型权重。运行前应检查配置的预训练选项及路径。完整环境与限制见 [运行说明](../../docs/runtime.md)。
