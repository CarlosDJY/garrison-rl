#!/bin/bash
# Before the sp3 training: sp2 on each given 盟约 (three 策略 mixed, 规程 14 and 20), then the training.
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTHONIOENCODING=utf-8
until grep -q "== runs/sp2/last.pt band_bonus" runs/sp2_band_eval.out; do sleep 10; done; sleep 5
for f in kjeragShip ursusShip lethaniaShip egirShip victoriaShip sargonShip; do
  echo "### focus $f"
  /c/Users/1/.conda/envs/asp-garrison/python.exe sp_band_eval.py --ckpt runs/sp2/last.pt --bands any --levels 14,20 --runs 128 --extra="--sp:bands=band_amiya,band_takecare,band_bonus;--sp:focus=$f" 2>&1 | grep -v "Warn\|TransformerEncoder"
done > runs/sp2_focus_eval.out 2>&1
bash runs/run_sp3.sh > runs/sp3.out 2>&1
