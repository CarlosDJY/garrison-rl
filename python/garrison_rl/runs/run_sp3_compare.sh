#!/bin/bash
# sp2 against sp3 on the same runs: three 策略 and six given 盟约 mixed (256 runs a level), then sp3 on each 盟约 alone.
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTHONIOENCODING=utf-8
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
B="--sp:bands=band_amiya,band_takecare,band_bonus"
F="kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip"
$PY sp_band_eval.py --ckpt runs/sp2/last.pt,runs/sp3/last.pt,runs/sp3/best.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="$B;--sp:focus=$F" --out runs/sp3_compare.json 2>&1 | grep -v "Warn\|TransformerEncoder"
for f in kjeragShip ursusShip lethaniaShip egirShip victoriaShip sargonShip; do
  echo "### focus $f"
  $PY sp_band_eval.py --ckpt runs/sp3/last.pt --bands any --levels 14,20 --runs 128 --extra="$B;--sp:focus=$f" 2>&1 | grep -v "Warn\|TransformerEncoder"
done
