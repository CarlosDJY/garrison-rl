#!/bin/bash
# sp1: PPO on Stronghold-Protocol's simulator (规程 1-20's six training levels, every fight the real one), from the
# policy that imitates the game's own AI (runs/sp_bc), held near it by a constant KL anchor. The policy owns the shop;
# the game AI lays out the field, equips, uses the 指令 and picks 机变 cards. Rewards: survive, HP, clear, leader share,
# and usable layers gained (--r-layer). On an abnormal exit, resume from state.pt (up to 20 times).
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 ASP_SIM=sp ASP_SP_PER_PROC=4
ARGS="--sim sp --sp-rehearsal 0 --levels 4,8,11,14,16,20 --open-first 6 --lr-warmup 10 --envs 96 --steps 128 --minibatch 256 --accum 16 --epochs 3 --lr 1e-4 --entropy 0.003 --kl-anchor 0.01 --anchor-refresh 0 --shape-plan 0 --shape-pot 0 --shape-dps 0 --shape-layers 0 --shape-plan-layers 0 --r-plan 0 --r-layer 0.3 --bc 0 --value-warmup 10 --updates 300 --refresh-every 0 --eval-every 20 --eval-runs 64 --real-every 0 --max-mem-fails 120 --init runs/sp_bc/last.pt --tag sp1"
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
$PY train_mix.py $ARGS && exit 0
for k in $(seq 1 20); do
  echo "=== restart $k at $(date)"
  sleep 120
  [ -f runs/sp1/state.pt ] || { echo "no state.pt yet: starting over"; $PY train_mix.py $ARGS && exit 0; continue; }
  $PY train_mix.py $ARGS --resume && exit 0
done
