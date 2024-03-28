# /bin/bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
    main_mlc.py --cfg config/SALGL/nuswide.yaml \
    --output checkpoint/salgl/nuswide/work1/ \
    --gpus 4,5,6,7 \
    --seed 42 \
    --print-freq 400
