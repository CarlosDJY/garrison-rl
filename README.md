# garrison-rl

用强化学习训练明日方舟「卫戍协议：规程」单人模式（规程等级 1–20）的运营策略：每回合在调度中心买谁、卖谁、何时刷新和升级。

对局环境基于 [Stardust-minus/Stronghold-Protocol](https://github.com/Stardust-minus/Stronghold-Protocol) 的模拟器。本仓库只包含训练代码和连接模拟器的环境脚本，不包含模拟器本体，也不包含任何游戏数据或素材。

这是一个进行中的个人实验项目，代码是边试边改的状态。

## 分工

- **策略网络负责**：调度中心里的决策（购买、出售、刷新、升级、拿晋升奖励、买装备、结束回合）。
- **模拟器自带 AI 负责**：站位、装备分配、策略（指令）的主动技能、机变。
- **环境指定**：每局的策略（开局选的那个）和主/副盟约。指定的盟约会写进观测，自带 AI 排阵时也会照顾它的成员。

## 目前最好的模型：sp3

到目前为止，后续的几轮实验（sp4–sp7）都没有超过 sp3，所以它仍是基线。

**它是怎么来的**

1. `sp_bc`：录制模拟器自带 AI 的对局，模仿它的调度中心决策，得到起点。
2. `sp2`：从 `sp_bc` 出发做 PPO，300 次更新。策略由自带 AI 自选，不指定盟约。
3. `sp3`：从 `sp2` 继续 150 次更新，改了两点：
   - 策略固定为阿米娅、华法琳、二重奏三个被动型策略，每局均匀抽一个。
   - 每局开局指定一个主盟约（谢拉格、乌萨斯、莱塔尼亚、阿戈尔、维多利亚、萨尔贡，均匀抽）。指定的盟约写进观测，自带 AI 排阵时优先照顾它的成员，局末按拿到的成员数和精锐数给一小笔奖励。

训练设置：规程等级 4 / 8 / 11 / 14 / 16 / 20 混合，96 个并行环境，每次更新每个环境 128 步，学习率 1e-4。完整命令见 `python/garrison_rl/runs/run_sp3.sh`。

**奖励**

- 每场战斗存活 +0.1，按损失的生命比例扣分，死亡 −2。
- 首领战按打掉的血量比例给分（最多 +5），击杀 +2，通关 +5。
- 每场战斗按本场新增的盟约层数给一小笔奖励（系数 0.3）。

**通关率**

打掉第 14 回合首领算通关，每档 256 局，三个策略均匀抽：

| 模型与盟约指定方式 | 规程 11 | 14 | 16 | 20 |
|---|---|---|---|---|
| 模拟器自带 AI（自选策略，参照） | 70% | 40% | 23% | 10% |
| sp2，不指定盟约 | 84% | 71% | 42% | 24% |
| sp3，主盟约均匀抽 | 86% | 76% | 50% | 32% |
| sp3，按首领取历史胜率最高的主盟约和小盟约 | 91% | 80% | 54% | 41% |
| sp3，同上，策略固定为二重奏（512 局） | — | — | — | 44% |

- 256 局的误差约 ±6 个点，512 局约 ±4 个点。
- 最后两行用到一张"首领 × 盟约"的胜率表，它是 sp5 训练过程中统计出来的（`train_mix.py --sp-draw 1`），放在 `python/garrison_rl/tables/sp5_draw_final.json`。
- 首领在开局就已知，所以"按首领选盟约"在实际对局里是可以做到的选择。

**已知的差距**

- 叠层不够。规程 20 到首领时最高的三个盟约约为 190 / 80 / 35 层，模拟器里能做到 450–550 层。
- 首领差别很大。规程 20 下矩兽约 3%、茧约 5%，这两个占了两成对局；胄、鳍、铳在 40% 以上。
- 规程 16 起约三成的局活不到首领。

## 做过的探索

每一轮的启动命令在 `python/garrison_rl/runs/`，训练记录在 `reports/playground/`。通关率是规程 11 / 14 / 16 / 20，256 局，按首领取最优盟约，可以直接和 sp3 的 91 / 80 / 54 / 41 比。

| 实验 | 改动 | 结果 |
|---|---|---|
| sp4 | 局末按最高几个盟约的层数给奖励，加入规程 18 | 没有超过 sp3 |
| sp5 | 训练时按首领的历史胜率抽主盟约和一个小盟约 | 90 / 78 / 51 / 39，持平。副产品是上面用到的胜率表 |
| sp6 | 每场战斗的叠层奖励从 0.3 提到 1.5 | 88 / 82 / 54 / 36，持平。小盟约层数从约 104 升到 146，通关率没变 |
| sp7 | 策略池加入克莱门莎，并规定它只玩阿戈尔（`--sp-band-focus`） | 原三个策略 90 / 79 / 54 / 40，持平。克莱门莎自己 78 / 59 / 42 / 26，训练后仍是最弱的策略 |

不训练、只分析现有对局记录得到的结论：

- **策略**：规程 16–20 的四万多局里，二重奏比阿米娅、华法琳高 2–4 个点，十个首领里八个它是第一或并列第一。按首领分别选策略没有额外收益（用一半数据选、另一半验证，不如固定二重奏）。
- **叠层节奏**：约一半的局在第 6–7 回合才激活主盟约，第二、第三盟约要到第 8–10 回合以后。模型很少用空弦、罗比菈塔这类在休整期叠层的干员（上场率 5–24%）。提高叠层奖励只让它稍微多留一点罗比菈塔。
- **茧**：赢法明确。主盟约莱塔尼亚时打到首领的局胜率 22%，其他主盟约 4–16%；其中配二重奏 33%，莱塔尼亚叠到 200 层以上 28–41%。
- **矩兽**：最好的主盟约是乌萨斯，也只有 10%；即使叠到 200 层以上且有 5 个精锐，也只有 25%。看起来不是调度中心决策能解决的。
- **赢局和输局的差别**：人数、精锐数、商店等级几乎一样，差别主要在层数（赢局约 220 层，输局约 180 层）。

小结：调整奖励、指定盟约、增加策略这三类改动都停在规程 20 约 40% 上下。靠选择（固定二重奏加按首领选盟约）能到 44% 左右。再往上需要解决弱首领，尤其是茧和矩兽。

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
| `python/garrison_rl/tables/` | 评估用的"首领 × 盟约"胜率表（训练中统计的对局胜负，不含游戏数据） |
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
| `sp3.pt` | 在 sp2 上继续 150 次更新，训练时固定三个策略并指定主盟约。目前最好的模型 |

存档只和训练时的模拟器版本（上面的提交号）配套：干员、盟约、装备的编号顺序来自那个版本。使用前先生成一份目录文件，再评估：

```bash
cd python/garrison_rl
python sp_spec.py                      # 写出 runs/sp_bc/spec.json
python sp_band_eval.py --ckpt path/to/sp3.pt --bands band_amiya,band_takecare,band_bonus \
  --levels 14,20 --runs 128 --extra="--sp:focus=kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip"
```

`--ckpt` 的路径相对于 `python/garrison_rl`。sp3 是按"给定主盟约"训练的，评估时要带上 `--sp:focus`；sp2 和 sp_bc 不需要。

上面的命令是主盟约均匀抽。要按首领取胜率最高的主盟约和小盟约（结果表里 41% 的那一行），加上小盟约的候选和胜率表：

```bash
python sp_band_eval.py --ckpt path/to/sp3.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000   --extra="--sp:bands=band_amiya,band_takecare,band_bonus;--sp:focus=kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip;--sp:second=docShip,steadShip,arcaneShip,preciShip,deputShip,skillfulShip,swiftShip,indomShip"   --draw tables/sp5_draw_final.json --best 1
```

把 `--bands any` 和 `--sp:bands=...` 换成 `--bands band_bonus` 就是固定二重奏。

## 许可与声明

- 本仓库以 GPL-3.0 发布（见 `LICENSE`）。环境脚本直接调用 Stronghold-Protocol 的代码，后者同样是 GPL-3.0。
- 《明日方舟》及「卫戍协议」的全部游戏内容归上海鹰角网络科技有限公司所有。本项目是非官方的爱好者研究项目，与鹰角网络没有任何关联，不包含游戏数据或素材。
- 模拟器的数值和机制以 Stronghold-Protocol 的实现为准，与实机可能存在差异，这里的通关率不代表实机表现。
