#!/bin/bash
# After sp4: its last checkpoint on the runs sp2 / sp3 were compared on (runs/sp3_compare.out).
cd /d/Code/Games/Arknights/asp-assistant-arknights/python/garrison_rl
export PYTHONIOENCODING=utf-8
until ! ps -ef | grep -v grep | grep -q "run_sp4.sh"; do sleep 20; done
sleep 10
/c/Users/1/.conda/envs/asp-garrison/python.exe sp_band_eval.py --ckpt runs/sp4/last.pt --bands any --levels 11,14,16,20 --runs 256 --seed 40000 --extra="--sp:bands=band_amiya,band_takecare,band_bonus;--sp:focus=kjeragShip,ursusShip,lethaniaShip,egirShip,victoriaShip,sargonShip" --out runs/sp4_compare.json > runs/sp4_compare.out 2>&1
