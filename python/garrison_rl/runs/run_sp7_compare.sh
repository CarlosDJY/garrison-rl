#!/bin/bash
# After sp7: the runs sp3 / sp5 / sp6 were compared on (seed 40000, 256 a level), main + second 盟约 from sp5's final
# win-rate table, each leader's best. Per 策略, so 克莱门莎 (always 阿戈尔) is listed apart from the three sp3 knows.
# Launch only once training has ended (no waiting here).
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTHONIOENCODING=utf-8
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
F="--sp:focus=kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip"
S="--sp:second=docShip,steadShip,arcaneShip,preciShip,deputShip,skillfulShip,swiftShip,indomShip"
G='Warn\|TransformerEncoder'
echo "### sp7 on the three old 策略 (same runs as sp3 91/80/54/41)"
$PY sp_band_eval.py --ckpt runs/sp7/last.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="--sp:bands=band_amiya,band_takecare,band_bonus;$F;$S" --draw runs/sp5/draw_final.json --best 1 --out runs/sp7_compare_best.json 2>&1 | grep --line-buffered -v "$G"
echo "### sp7 on 克莱门莎 / 阿米娅, main = 阿戈尔 (same runs as sp3 68/30 and 77/34)"
$PY sp_band_eval.py --ckpt runs/sp7/last.pt --bands band_clementia,band_amiya --levels 14,20 --runs 128 --seed 41000 --extra="--sp:focus=egirShip;$S" --draw runs/sp5/draw_final.json --best 1 --out runs/sp7_newbands_b.json 2>&1 | grep --line-buffered -v "$G"
echo "### sp7 on 克莱门莎 only, main = 阿戈尔, all four levels"
$PY sp_band_eval.py --ckpt runs/sp7/last.pt --bands band_clementia --levels 11,14,16,20 --runs 256 --seed 40000 --extra="--sp:focus=egirShip;$S" --draw runs/sp5/draw_final.json --best 1 --out runs/sp7_clementia.json 2>&1 | grep --line-buffered -v "$G"
echo "### done"
