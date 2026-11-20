#!/bin/bash
# sp4: sp3 trained on for the high levels' stacking. 规程 18 joins (16-20 share their normal rounds, only the
# leader's HP rises: a step between 16 and 20), and --r-stack 3 pays at the run's end for the four highest 盟约's layers
# ((l1 / 500 + l2 / 300 + l3 / 200 + l4 / 200) / 3, no cap: 3 at the user's own 规程 20 runs' 500 / 300 / 200 at the
# 隐秘核心, 4.2 at a strong player's 600 / 300 / 200 / 200; the policy's clears are at about 200 / 70 / 30, 0.8). Otherwise
# sp3's settings (three 策略, six given 盟约). On an abnormal exit, resume.
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 ASP_SIM=sp ASP_SP_PER_PROC=4
ARGS="--sim sp --sp-rehearsal 0 --sp-boss-rehearsal 3 --sp-bands band_amiya,band_takecare,band_bonus --sp-focus kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip --levels 4,8,11,14,16,18,20 --open-first 7 --lr-warmup 5 --envs 96 --steps 128 --minibatch 256 --accum 16 --epochs 3 --lr 1e-4 --entropy 0.003 --kl-anchor 0.01 --anchor-refresh 10 --shape-plan 0 --shape-pot 0 --shape-dps 0 --shape-layers 0 --shape-plan-layers 0 --r-plan 1 --r-layer 0.3 --bc 0 --value-warmup 5 --r-stack 3 --updates 150 --refresh-every 0 --eval-every 30 --eval-runs 64 --real-every 0 --max-mem-fails 120 --init runs/sp3/last.pt --init-critic runs/sp3/critic_last.pt --tag sp4 --sp-fights data/sp_train/sp4"
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
$PY train_mix.py $ARGS && exit 0
for k in $(seq 1 20); do
  echo "=== restart $k at $(date)"; sleep 120
  $PY train_mix.py $ARGS --resume && exit 0
done
