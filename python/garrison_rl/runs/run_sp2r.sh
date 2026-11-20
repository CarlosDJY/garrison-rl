#!/bin/bash
# sp2, resumed: the same run as run_sp2.sh from its state.pt, with the simulator processes stepping apart (train_mix.py
# --async-steps, the default) and every real fight of the training envs kept for the fight model (data/sp_train/sp2_*).
# On an abnormal exit, resume again (up to 20 times).
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTHONIOENCODING=utf-8 ASP_SIM=sp ASP_SP_PER_PROC=4
ARGS="--sim sp --sp-rehearsal 0 --sp-boss-rehearsal 3 --levels 4,8,11,14,16,20 --open-first 6 --lr-warmup 10 --envs 96 --steps 128 --minibatch 256 --accum 16 --epochs 3 --lr 1e-4 --entropy 0.003 --kl-anchor 0.01 --anchor-refresh 0 --shape-plan 0 --shape-pot 0 --shape-dps 0 --shape-layers 0 --shape-plan-layers 0 --r-plan 0 --r-layer 0.3 --bc 0 --value-warmup 10 --updates 300 --refresh-every 0 --eval-every 30 --eval-runs 64 --real-every 0 --max-mem-fails 120 --init runs/sp_bc/last.pt --tag sp2 --sp-fights data/sp_train/sp2"
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
for k in $(seq 0 20); do
  [ $k -gt 0 ] && { echo "=== restart $k at $(date)"; sleep 120; }
  $PY train_mix.py $ARGS --resume && exit 0
done
