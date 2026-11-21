#!/bin/bash
# After sp5: on the runs sp2 / sp3 were compared on (seed 40000, 256 a level, three 策略), with a main and a second 盟约
# given per leader from sp5's win-rate table: sp5 drawing by win rate, sp5 taking each leader's best, sp3 drawing.
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTHONIOENCODING=utf-8
PY=/c/Users/1/.conda/envs/asp-garrison/python.exe
until ! ps -ef | grep -v grep | grep -q "run_sp5.sh"; do sleep 15; done
sleep 5
X="--sp:bands=band_amiya,band_takecare,band_bonus;--sp:focus=kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip;--sp:second=docShip,steadShip,arcaneShip,preciShip,deputShip,skillfulShip,swiftShip,indomShip"
cp runs/sp5/draw.json runs/sp5/draw_final.json
echo "### sp5, drawn by win rate"
$PY sp_band_eval.py --ckpt runs/sp5/last.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="$X" --draw runs/sp5/draw_final.json 2>&1 | grep --line-buffered -v "Warn\|TransformerEncoder"
echo "### sp5, each leader's best"
$PY sp_band_eval.py --ckpt runs/sp5/last.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="$X" --draw runs/sp5/draw_final.json --best 1 2>&1 | grep --line-buffered -v "Warn\|TransformerEncoder"
echo "### sp3, drawn by win rate"
$PY sp_band_eval.py --ckpt runs/sp3/last.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="$X" --draw runs/sp5/draw_final.json 2>&1 | grep --line-buffered -v "Warn\|TransformerEncoder"
