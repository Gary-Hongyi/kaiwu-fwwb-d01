# 复赛：王者荣耀 1v1 最终提交源码

此目录来自 `代码.zip/代码/复赛代码/`，由项目作者确认是复赛最后选择提交的源码版本。整理时保留源码内容，只将缺失对应实现的 `agent_ppo/debug/` 检查脚本移至 [V2 试验归档](../../archives/hok1v1/v2_checks/)。

当前应用配置两方均使用 `ppo`，主路径如下：

```text
原始观测 → unpack_state_dict → obs_builder → Model（实体编码 + LSTM）
        → 动作采样与修复 → 环境交互 → reward_manager → 样本整理 → PPO 更新
```

- 结构化特征共 4348 维，包含英雄、小兵、防御塔、河蟹与飞行物等。
- 六个离散动作头大小为 `[12, 16, 16, 16, 16, 9]`。
- 价值头分为 `farm / kda / damage / push / state` 五组。
- 包含安全推塔、合法动作修复和召唤师技能选择等规则。
- 相比同源归档分支，此版本配置了 `kite_attack`、`low_hp_retreat`、`minion_shield` 三项额外奖励；没有完整实验记录证明每项奖励的实际收益。

在匹配的开悟环境内，以本目录为代码根目录运行：

```bash
python train_test.py
```

此脚本默认算法为 PPO，但它在找不到平台训练组件时会转入一个 **DIY 本地烟雾检查**。该回退路径通过时，只能说明备选 DIY 的导入、推理和样本结构检查通过，不能证明 PPO 或游戏训练已通过。

`kaiwu.json` 中的模型池编号 `289614` 按提交源码保留，属于原平台配置；在其他账号或环境运行前需确认可访问性。`conf/configure_app.toml` 默认关闭预训练加载。本目录未附带与最终提交源码严格对应的模型。

`agent_diy/`、旧 `feature_process/` 及旧 `reward_process.py` 保留作代码背景；当前 PPO 使用的是 `obs_builder.py` 和 `reward_manager.py`。

另一个现存分支见 [regional_branch](../../archives/hok1v1/regional_branch/)。不要直接把它的加载参数或权重视为本提交版本的组成部分。
