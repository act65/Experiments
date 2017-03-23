set -x TRAIL_NAME filters16_10
python hyper.py  --logdir=/media/ws_sc_2/Storage/Experiments/hyper/$TRAIL_NAME/vanilla --hyper=False
python hyper.py  --logdir=/media/ws_sc_2/Storage/Experiments/hyper/$TRAIL_NAME/hyper --hyper=True
