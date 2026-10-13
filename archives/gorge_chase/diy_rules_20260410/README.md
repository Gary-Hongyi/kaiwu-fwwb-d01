# 初赛 DIY 规则快照

来源为 `gorge_chase_diy_model_eval_ckptroot_20260410.zip` 中的外层 `ckpt/`。该外层目录实际是代码根，整理时展开到了当前目录；内层权重目录未收录。原始说明保留在 `README_RULES.md`。

主要逻辑位于 `agent_diy/rule_policy.py`，根据追击怪物、宝箱、加速、通行空间、陷阱和合法动作信息评分。`train_test.py` 默认选择 `diy`。

在匹配的开悟环境中，以此目录为代码根运行 `python train_test.py`。本地可执行原有场景示例：

```bash
python -m agent_diy.self_test_rule_policy
```

该脚本打印若干场景的动作与概率，不是完整游戏评测。保留的 `conf/prompt.py` 是提示词辅助文本，不能据此认定本方案实际调用了大模型；其中追怪措辞与躲避追击任务可能不一致，整理时没有改写历史文件。
