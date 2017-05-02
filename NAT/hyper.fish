set -x TRAIL_NAME $argv[1]
python hyper.py  --logdir=/media/ws_sc_2/Storage/Experiments/hyper/$TRAIL_NAME --hyper=True --lr=$argv[2] --beta=$argv[3]
python hyper.py  --logdir=/media/ws_sc_2/Storage/Experiments/hyper/$TRAIL_NAME --hyper=False --lr=$argv[2] --beta=$argv[3]
