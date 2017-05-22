set TRAIL_NAME $argv[1]
set LR $argv[2]
set BATCH $argv[3]
python train.py  --logdir=/media/ws_sc_2/Storage/Generalisation/$TRAIL_NAME/lr-$LR/batchsize-$BATCH/deep/0 --lr=$LR --deep=True --batchsize=$BATCH
python train.py  --logdir=/media/ws_sc_2/Storage/Generalisation/$TRAIL_NAME/lr-$LR/batchsize-$BATCH/wide/0 --lr=$LR --deep=False --batchsize=$BATCH
python train.py  --logdir=/media/ws_sc_2/Storage/Generalisation/$TRAIL_NAME/lr-$LR/batchsize-$BATCH/deep/1 --lr=$LR --deep=True --batchsize=$BATCH
python train.py  --logdir=/media/ws_sc_2/Storage/Generalisation/$TRAIL_NAME/lr-$LR/batchsize-$BATCH/wide/1 --lr=$LR --deep=False --batchsize=$BATCH
python train.py  --logdir=/media/ws_sc_2/Storage/Generalisation/$TRAIL_NAME/lr-$LR/batchsize-$BATCH/deep/2 --lr=$LR --deep=True --batchsize=$BATCH
python train.py  --logdir=/media/ws_sc_2/Storage/Generalisation/$TRAIL_NAME/lr-$LR/batchsize-$BATCH/wide/2 --lr=$LR --deep=False --batchsize=$BATCH
