#!/bin/bash
# sp3: sp2 trained on with (1) the 策略 fixed to three passive ones (阿米娅 / 华法琳 / 二重奏, one per match, evenly) and
# (2) every match given one of six core 盟约 as its main, evenly (谢拉格 / 乌萨斯 / 莱塔尼亚 / 阿戈尔 / 维多利亚 / 萨尔贡: the
# plan the policy sees, the main the layout works for, --r-plan 1 at the run's end for its members held / 精锐), so that
# each 盟约 is tried as often as the others instead of following the game AI's rule pick. The KL anchor moves (every
# 10 updates). Policy and critic start from sp2's. On an abnormal exit, resume.
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 ASP_SIM=sp ASP_SP_PER_PROC=4
ARGS="--sim sp --sp-rehearsal 0 --sp-boss-rehearsal 3 --sp-bands band_amiya,band_takecare,band_bonus --sp-focus kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip --levels 4,8,11,14,16,20 --open-first 6 --lr-warmup 5 --envs 96 --steps 128 --minibatch 256 --accum 16 --epochs 3 --lr 1e-4 --entropy 0.003 --kl-anchor 0.01 --anchor-refresh 10 --shape-plan 0 --shape-pot 0 --shape-dps 0 --shape-layers 0 --shape-plan-layers 0 --r-plan 1 --r-layer 0.3 --bc 0 --value-warmup 5 --updates 150 --refresh-every 0 --eval-every 30 --eval-runs 64 --real-every 0 --max-mem-fails 120 --init runs/sp2/last.pt --init-critic runs/sp2/critic_last.pt --tag sp3 --sp-fights data/sp_train/sp3"
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
$PY train_mix.py $ARGS && exit 0
for k in $(seq 1 20); do
  echo "=== restart $k at $(date)"; sleep 120
  $PY train_mix.py $ARGS --resume && exit 0
done
