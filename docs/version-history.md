# 现存版本关系

本仓库于 2026-10-13 进行赛后整理。比赛期间没有完整 Git 记录，一些中间迭代源码、实验日志和精确版本对应关系没有保留下来。以下同时区分作者确认的信息与文件比对得到的信息。

## 作者确认

1. 初赛是峡谷追逐，复赛是智能体 1v1。
2. `代码.zip` 中的复赛代码与原“区域赛提交版”是从复赛某次迭代分出的不同分支。
3. 最终复赛提交选择了 `代码.zip` 中的版本。
4. 共同分支点及部分中间迭代源码已缺失。
5. 参赛代码当时在腾讯提供的官方参赛平台 Dev 环境中运行和迭代。

因此保留两份源码，未合并为一个新算法版本。Git 历史只记录本次整理及以后的工作，不伪造比赛期间的提交、日期或标签。

## 留存版本清单

| 阶段 | 仓库路径 | 留存来源 | 定位 |
|---|---|---|---|
| 初赛 | `projects/gorge_chase` | 代码.zip 中初赛代码的 flash优化副本/code | 现存 PPO 方案，精确迭代编号未确认 |
| 初赛 | `archives/gorge_chase/diy_rules_20260410` | gorge_chase_diy_model_eval_ckptroot_20260410.zip | DIY 规则方案，原 README 标记 rules_v1.0 |
| 初赛 | `baselines/gorge_chase` | code-gorge_chase-public-15.0.1-comp-normal-lite.26comp.zip | 平台基线 |
| 复赛 | `projects/hok1v1` | 代码.zip/代码/复赛代码 | 作者确认的最终提交源码 |
| 复赛 | `archives/hok1v1/regional_branch` | 原区域赛提交版的 agent_ppo、agent_diy、conf | 同源的另一条迭代分支 |
| 复赛 | `baselines/hok1v1` | _codex_tmp/baseline1_20260602_230412 | 基线快照 |
| 复赛试验 | `archives/hok1v1/v2_checks` | 代码.zip 中 agent_ppo/debug | 仅保留检查脚本，缺少实现 |

`rule-v5(3).zip` 的元数据显示为另一个赛题 `back_to_the_realm_v2`，未纳入此次仓库。原包继续保存在本地。

## 复赛两条分支的差异

原始 ZIP 的复赛部分有 57 个非缓存文件：与原区域赛目录相比，45 个逐字节一致、8 个不同、4 个仅 ZIP 有。区域赛对应代码、配置、权重范围另有 3 个 ckpt 文件。这个比较是在整理和移动 V2 脚本之前完成的。

| 内容 | 最终提交版：代码.zip | 同源分支：原区域赛提交版 |
|---|---|---|
| 微操奖励 | kite_attack=0.3、low_hp_retreat=0.4、minion_shield=0.2，并接入价值分组 | 无上述三项 |
| 预训练 | 默认关闭，预留编号 1000 | 默认开启，编号 276912 |
| 显式评估配置 | configure_app.toml 未指定评估模型路径/编号 | 指向编号 292449 |
| 根文件 | train_test.py、kaiwu.json、.gitignore、.vscode/launch.json 齐全 | 缺上述四个文件 |
| 权重 | 复赛子目录没有附权重 | 附第 292449 步权重与元数据 |
| 监控标签 | 智能决策1v1、特征统计 | 标签带 Taichu-2025，面板英文标识也不同 |

8 个不同文件：

- `agent_ppo/conf/conf.py`、`agent_ppo/feature/reward_manager.py`：奖励配置和计算逻辑差异。
- `conf/configure_app.toml`：模型加载和评估配置差异。
- `agent_ppo/conf/monitor_builder.py`：监控标签、标识和注释差异。
- `agent_ppo/algorithm/algorithm.py`、`agent_ppo/feature/obs_builder.py`、`agent_ppo/feature/unpack_state_dict.py`、`agent_ppo/utils/frame_dump.py`：注释/文档字符串差异；去掉文档字符串后 AST 相同。

网络、Agent、训练工作流及样本定义等主要文件一致。`batchsize2_292449.zip` 中 56 个非缓存文件与原区域赛目录的对应文件完全一致；它是该分支的现存快照，不应据此改称最终提交源码。

第 292449 步权重对应的全部训练阶段与奖励设置无法从现存资料还原；没有验证它是否能直接作为最终提交版的配套模型，也不以此宣称最终提交版的评测结果。

## 初赛 PPO 与平台基线

原初赛 PPO 代码与平台基线比对，29 个文件相同，8 个文件不同：PPO 的 algorithm、conf、monitor_builder、train_env_conf、preprocessor、model、train_workflow，以及全局 configure_app.toml。现存实现增加了局部空间 CNN、访问/探索记忆、阶段奖励和闪现相关处理，但没有完整实验记录可重建这些改动的先后与收益。
