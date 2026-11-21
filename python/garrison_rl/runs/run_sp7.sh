#!/bin/bash
# sp7: from sp3, sp5 settings (given main + second drawn by win rate, r-stack 3, r-layer 0.3, level 18), plus the 策略 克莱门莎
# (twice in the list: ~20 % of the matches). A 克莱门莎 match always has 阿戈尔 as its main; when the match has 阿戈尔
# off, one of the other three 策略 is played instead. Its results stay out of the win-rate table.
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 ASP_SIM=sp ASP_SP_PER_PROC=4
ARGS="--sim sp --sp-rehearsal 0 --sp-boss-rehearsal 3 --sp-bands band_amiya,band_takecare,band_bonus,band_clementia,band_clementia --sp-band-focus band_clementia=egirShip --sp-focus kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip --sp-second docShip,steadShip,arcaneShip,preciShip,deputShip,skillfulShip,swiftShip,indomShip --sp-draw 1 --levels 4,8,11,14,16,18,20 --open-first 7 --lr-warmup 5 --envs 96 --steps 128 --minibatch 256 --accum 16 --epochs 3 --lr 1e-4 --entropy 0.003 --kl-anchor 0.01 --anchor-refresh 10 --shape-plan 0 --shape-pot 0 --shape-dps 0 --shape-layers 0 --shape-plan-layers 0 --r-plan 1 --r-layer 0.3 --bc 0 --value-warmup 5 --r-stack 3 --updates 150 --refresh-every 0 --eval-every 30 --eval-runs 64 --real-every 0 --max-mem-fails 120 --init runs/sp3/last.pt --init-critic runs/sp3/critic_last.pt --tag sp7 --sp-fights data/sp_train/sp7"
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
$PY train_mix.py $ARGS && exit 0
for k in $(seq 1 20); do
  echo "=== restart $k at $(date)"; sleep 120
  $PY train_mix.py $ARGS --resume && exit 0
done
