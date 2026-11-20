# garrison-rl

用强化学习训练明日方舟「卫戍协议：规程」单人模式（规程等级 1–20）的运营策略：每回合在调度中心买谁、卖谁、何时刷新和升级。

对局环境基于 [Stardust-minus/Stronghold-Protocol](https://github.com/Stardust-minus/Stronghold-Protocol) 的模拟器。本仓库只包含训练代码和连接模拟器的环境脚本，不包含模拟器本体，也不包含任何游戏数据或素材。

这是一个进行中的个人实验项目，代码是边试边改的状态。

## 分工

- **策略网络负责**：调度中心里的决策（购买、出售、刷新、升级、拿晋升奖励、买装备、结束回合）。
- **模拟器自带 AI 负责**：站位、装备分配、策略（指令）的主动技能、机变。
- **环境指定**：每局的策略（开局选的那个）和主/副盟约。指定的盟约会写进观测，自带 AI 排阵时也会照顾它的成员。

## 目前的结果

规程等级 11 / 14 / 16 / 20 的通关率（打掉第 14 回合首领即算通关），每档 256 局，策略固定为阿米娅、华法琳、二重奏三选一，主盟约在谢拉格、乌萨斯、莱塔尼亚、阿戈尔、维多利亚、萨尔贡中均匀指定：

| 模型 | 11 | 14 | 16 | 20 |
|---|---|---|---|---|
| sp2（从模仿自带 AI 的起点做 PPO，300 次更新） | 84% | 71% | 42% | 24% |
| sp3（在 sp2 上加指定盟约，150 次更新） | 86% | 76% | 50% | 32% |

作为参照，模拟器自带 AI 自己打（自选策略）约为 70% / 40% / 23% / 10%。

已知的主要差距：

- 叠层不够。规程 20 的通关局到首领时三个最高盟约约 200 / 70 / 30 层，而手打能到 500 / 300 / 200。
- 首领差别很大。矩兽、茧几乎全输，昆图斯很依赖盟约（阿戈尔明显更好）。
- 规程 16 起约三成的局活不到首领。

sp4、sp5 是针对这些问题的后续实验（叠层奖励、规程 18、按首领的历史胜率抽主盟约和小盟约），结果见 `reports/playground/`。

## 目录

| 路径 | 内容 |
|---|---|
| `sp-gym/spgym.mjs` | 环境：用 Node 跑 Stronghold-Protocol 的对局，通过标准输入输出和训练程序交换 JSON |
| `sp-gym/patches/` | 需要给 Stronghold-Protocol 打的一行补丁（导出两个内部函数） |
| `python/garrison_rl/train_mix.py` | 多难度混合的 PPO 训练 |
| `python/garrison_rl/model.py` | 观测编码和策略网络 |
| `python/garrison_rl/gym_client.py` | 环境客户端（多进程，各进程独立步进） |
| `python/garrison_rl/sp_record.py`、`sp_bc.py` | 录制自带 AI 的决策并做模仿学习，得到训练起点 |
| `python/garrison_rl/sp_band_eval.py` | 固定策略 / 盟约的评估 |
| `python/garrison_rl/sp_fight.py`、`sp_layout.py` | 战斗结果预测模型和站位选择模型（实验性，目前没有接入训练） |
| `python/garrison_rl/runs/run_sp*.sh` | 每一轮实验实际使用的启动命令 |
| `reports/playground/` | 各轮实验的训练记录和分析 |
| `tools/sync_from_dev.sh` | 从开发目录同步文件到本仓库 |

`python/garrison_rl` 里不带 `sp_` 前缀的其余脚本（`fight_model.py`、`gen_fights.py`、`train_ppo.py` 等）来自更早的阶段，那时用的是另一套自写的 C# 环境，不在本仓库里；其中 `train_ppo.py`、`model.py` 仍被现在的训练引用。

## 运行

需要：

- Node 24 或更高（Stronghold-Protocol 的要求）。
- Python 3.11、PyTorch 2.x（带 CUDA）、NumPy。
- Stronghold-Protocol 的一份克隆，放在本仓库根目录下并打上补丁。开发时用的是提交 `9a41bbe`：

```bash
git clone https://github.com/Stardust-minus/Stronghold-Protocol.git
cd Stronghold-Protocol
git checkout 9a41bbe98255a70cf874814245493fa98cf019bb
git apply ../sp-gym/patches/0001-export-prep-preamble.patch
```

无需 `npm install`，无界面运行不依赖第三方包。

路径通过环境变量指定：

```bash
export ASP_SIM=sp
export ASP_SP_NODE=/path/to/node
export ASP_SP_GYM=/path/to/garrison-rl/sp-gym/spgym.mjs
export ASP_SP_PER_PROC=4
```

之后的流程是：`sp_record.py` 录制自带 AI 的对局，`sp_bc.py` 模仿学习得到起点，`train_mix.py --sim sp ...` 做 PPO。具体参数见 `python/garrison_rl/runs/run_sp*.sh`。这些启动脚本里写的是开发机上的绝对路径，使用前需要改成自己的。

## 训练好的模型

模型存档放在 [Releases](https://github.com/CarlosDJY/garrison-rl/releases) 页面，不在代码仓库里。每个文件约 10 MB，是策略网络的权重。

| 文件 | 说明 |
|---|---|
| `sp_bc.pt` | 模仿模拟器自带 AI 得到的起点 |
| `sp2.pt` | 在起点上做 PPO，300 次更新。训练时策略由自带 AI 自选，没有指定盟约 |
| `sp3.pt` | 在 sp2 上继续 150 次更新，训练时固定三个策略并指定主盟约。上面结果表里的 sp3 |

存档只和训练时的模拟器版本（上面的提交号）配套：干员、盟约、装备的编号顺序来自那个版本。使用前先生成一份目录文件，再评估：

```bash
cd python/garrison_rl
python sp_spec.py                      # 写出 runs/sp_bc/spec.json
python sp_band_eval.py --ckpt path/to/sp3.pt --bands band_amiya,band_takecare,band_bonus \
  --levels 14,20 --runs 128 --extra="--sp:focus=kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip"
```

`--ckpt` 的路径相对于 `python/garrison_rl`。sp3 是按"给定主盟约"训练的，评估时要带上 `--sp:focus`；sp2 和 sp_bc 不需要。

## 许可与声明

- 本仓库以 GPL-3.0 发布（见 `LICENSE`）。环境脚本直接调用 Stronghold-Protocol 的代码，后者同样是 GPL-3.0。
- 《明日方舟》及「卫戍协议」的全部游戏内容归上海鹰角网络科技有限公司所有。本项目是非官方的爱好者研究项目，与鹰角网络没有任何关联，不包含游戏数据或素材。
- 模拟器的数值和机制以 Stronghold-Protocol 的实现为准，与实机可能存在差异，这里的通关率不代表实机表现。
