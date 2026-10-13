# 运行环境与复现范围

## 完整训练

参赛期间，代码是在腾讯提供的官方参赛平台 Dev 环境中运行和迭代的。这一运行背景由项目作者确认。本次整理在普通本地电脑进行，没有重新接入原比赛平台；以下说明的是如何保留和理解平台依赖，不是据本地缺包推断原工程无法运行。

代码依赖腾讯开悟的游戏环境、训练组件及平台工具，包括 `kaiwudrl`、部分路径使用的 `kaiwu_agent`、`common_python`、平台提供的 `tools.eval` 等。它们未随本仓库分发。不要用同名的陌生第三方包替代这些依赖。

每次选择一个运行根目录，例如 `projects/gorge_chase` 或 `projects/hok1v1`，将其内容作为对应开悟项目的代码根。这里保留多个同名 `agent_ppo` 包，不能把所有运行根一起加入 PYTHONPATH。

启动短训练测试：

```bash
cd projects/hok1v1
python train_test.py
```

初赛同理，在 `projects/gorge_chase` 下运行。正式训练按平台工作流执行，并核对 `conf/configure_app.toml`，其参数与 train_test 中覆盖的参数不同。

初赛来源元数据记录环境为 `gorge_chase`、`15.0.1-comp-normal-lite.26comp`。复赛平台配置使用 `hok1v1`；另一分支的模型元数据记录版本 61.1.4，但这不能独立证明最终提交训练时的精确环境。未保存完整依赖锁文件，不提供未经确认的 requirements 锁定版本。

## 必须核对的原配置

- 复赛最终源码 `kaiwu.json` 的模型池编号为 289614，可能只在原平台上下文可访问，需按实际运行环境调整。
- 最终复赛版本默认 `preload_model=false`。另一分支默认加载 276912，并指向 292449 评估；对应权重不随仓库分发。
- 平台配置引用 `tools.eval.workflow` 和 `tools.action_space_1v1`，本仓库没有这些平台工具。是否自动提供需要在实际环境确认。
- 根目录的 `.vscode/launch.json` 按来源保留，需单独打开对应项目目录才与 `${workspaceFolder}/train_test.py` 对应。
- 归档的 regional_branch 在当前留存目录中缺少启动文件，保持现存快照状态，没有借用其他分支补齐；这不表示当时平台 Dev 工程也缺少这些文件。

## 普通电脑上的检查

文件与语法检查只需 Python 3.11+ 标准库：

```bash
python scripts/verify_repository.py
```

它检查来源文件 SHA-256、Python 语法、TOML 与 JSON 格式、配置引用的本地 Agent/workflow，并列出外部平台工具。它不加载权重、不启动环境，也不评价智能体性能。

如已具备 PyTorch 和 NumPy，可以做独立网络前向检查；本次实际执行结果见 validation.md。版本仅表示检查时的本地环境，不是原比赛依赖要求。

复赛 `train_test.py` 在缺少 kaiwudrl 时尝试执行 DIY 本地回退检查，这条路径不验证 PPO；如果其他依赖不满足，也可能无法运行。初赛启动入口直接依赖平台，不提供同样的本地回退。
