# 目录分类与文件命名依据

本仓库是赛后整理的资料库，先区分比赛阶段，再区分运行代码、基线、同源分支和答辩资料。原始目录 `D:/competition/fwwb` 保留，仓库内的复制件才重命名。每个保存的来源文件在 `docs/source_manifest.json` 中记录原路径和哈希。

| 目录 | 收录依据 | 分类原因 |
|---|---|---|
| `projects/gorge_chase/` | 初赛峡谷追逐的现存 PPO 工程 | 独立赛题与配置根目录 |
| `projects/hok1v1/` | 作者确认的复赛最终提交源码，来自 `代码.zip/复赛代码` | 保持平台导入路径与提交时工程结构 |
| `baselines/` | 能识别来源的官方或早期基线 | 基线与本队最终方案用途不同 |
| `archives/gorge_chase/diy_rules_20260410/` | 初赛另一套 DIY 规则快照 | 与初赛 PPO 实现方式不同 |
| `archives/hok1v1/regional_branch/` | 原“区域赛提交版”的同源分支 | 避免把两分支奖励、配置和模型合并 |
| `archives/hok1v1/v2_checks/` | 仅留存检查脚本的 V2 试验 | 对应实现缺失 |
| `docs/competition/regional-final-defense/` | 区域赛决赛答辩讲稿和 PPT | 演讲材料按使用场合归档 |
| `docs/competition/regional-technical-notes/` | 复赛“特征与规则”技术文稿 | 说明算法设计，但不是运行代码 |
| `docs/` 其他文件 | 仓库说明、版本关系、运行限制、验证与来源 | 由赛后整理写成，区别于比赛原件 |

英文目录 `gorge_chase` 和 `hok1v1` 与平台项目代码一致；中文材料用中文文件名说明赛事、主题和版本。`regional-final-defense` 明确指区域赛决赛答辩，避免笼统的 `ppt` 文件夹。

## 命名规则

1. 材料用“场景 + 内容 + 版本特点”。例如 `区域赛决赛答辩_赛题与展望_压缩稿.docx` 能看出场合、主题和区别。
2. 只写能确认的版本信息。原名 `2` 可标“第二稿”；`333` 原意未知，改为“变体 B”，不暗示时间先后。没有证据证明哪份 PPT 最终演示，所以没有“最终版”文件名。
3. 技术稿按原名的“草稿、终稿、复盘稿”命名，去掉原作者内部使用的章节序号和人名，把主题放在前面。
4. `agent.py`、`model/model.py`、`workflow/train_workflow.py`、`conf/*.toml` 等代码文件与平台导入或配置引用有关，保持原名；项目 README 和架构文档解释职责。
5. `hero_specific_feacures.py` 的拼写、`utils/_init_.py` 的非标准写法、`baseline1_20260602_230412` 的时间后缀属于历史源码与来源标识。直接改名会破坏来源比对或潜在引用；本次保留并在文档中说明。

已检查仓库中的源码、配置、文档、讲稿和 PPT 文件名。新整理的材料都按用途命名；同名的 `agent_ppo`、`agent_diy` 由各自项目或归档目录区分。`README.md`、`.gitignore`、`.gitattributes`、`train_test.py` 是约定的入口名。

六份 PPT 的原文件映射、页数和差别见[比赛资料索引](competition/document-index.md)。项目作者确认演讲人为郭鸿毅；三份 PPT 整理副本中的六处误写已更正，其他幻灯片内容保持原样。备赛会议记录与受损讲稿留在原目录，未据文件名推断为最终参赛材料。
