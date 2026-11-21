#!/bin/bash
# After sp6: the runs sp3 / sp5 were compared on (seed 40000, 256 a level, three 策略), main + second 盟约 from sp5's
# final win-rate table (the one sp3 scored 91/80/54/41 with): sp6 taking each leader's best, sp6 drawing.
# Then sp3 on two further passive 策略 (克莱门莎 with 阿戈尔 as its main, 伊奥莱塔·罗素), 128 runs a level.
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTHONIOENCODING=utf-8
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
until ! ps -ef | grep -v grep | grep -qE "run_sp6_resume|train_mix\.py"; do sleep 15; done
sleep 5
F="--sp:focus=kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip"
S="--sp:second=docShip,steadShip,arcaneShip,preciShip,deputShip,skillfulShip,swiftShip,indomShip"
X="--sp:bands=band_amiya,band_takecare,band_bonus;$F;$S"
G='Warn\|TransformerEncoder'
echo "### sp6, each leader's best"
$PY sp_band_eval.py --ckpt runs/sp6/last.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="$X" --draw runs/sp5/draw_final.json --best 1 --out runs/sp6_compare_best.json 2>&1 | grep --line-buffered -v "$G"
echo "### sp6, drawn by win rate"
$PY sp_band_eval.py --ckpt runs/sp6/last.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="$X" --draw runs/sp5/draw_final.json --out runs/sp6_compare_draw.json 2>&1 | grep --line-buffered -v "$G"
echo "### sp3 on 克莱门莎 / 阿米娅 (reference), main = 阿戈尔"
$PY sp_band_eval.py --ckpt runs/sp3/last.pt --bands band_clementia,band_amiya --levels 14,20 --runs 128 --seed 41000 --extra="--sp:focus=egirShip;$S" --draw runs/sp5/draw_final.json --best 1 --out runs/sp3_newbands_b.json 2>&1 | grep --line-buffered -v "$G"
echo "### done"
