set -x TRAIL filters16_10
python atrous.py  --logdir=/media/ws_sc_2/Storage/Experiments/atrous/$TRAIL/conv/noscale --scale=False --atrous=False
python atrous.py  --logdir=/media/ws_sc_2/Storage/Experiments/atrous/$TRAIL/conv/scale --scale=True --atrous=False
python atrous.py  --logdir=/media/ws_sc_2/Storage/Experiments/atrous/$TRAIL/atrous/noscale --scale=False --atrous=True
python atrous.py  --logdir=/media/ws_sc_2/Storage/Experiments/atrous/$TRAIL/atrous/scale --scale=True --atrous=True
