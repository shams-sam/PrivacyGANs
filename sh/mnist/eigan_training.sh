# A
python expt_mnist/eigan_training.py --device gpu --n-gpu 1 --n-ally 2 --n-advr 10 --batch-size 512 --n-epochs 101 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.001 --lr-advr 0.001 --alpha 0.5 --expt mnist --encd-ckpt ckpts/mnist/models/mnist_encd_pretrain_A.stop --ally-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_ally_0.stop --advr-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_advr_0.stop

# B: no
# python expt_mnist/eigan_training.py --device gpu --n-gpu 1 --n-ally 2 --n-advr 10 --batch-size 32 --n-epochs 50 --shuffle 0 --init-w 1 --lr-encd 0.01 --lr-ally 0.01 --lr-advr 0.01 --alpha 0.5 --expt mnist --encd-ckpt ckpts/mnist/models/mnist_encd_pretrain_A.stop --ally-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_ally_0.stop --advr-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_advr_0.stop

# C: no
# python expt_mnist/eigan_training.py --device gpu --n-gpu 1 --n-ally 2 --n-advr 10 --batch-size 32 --n-epochs 100 --shuffle 0 --init-w 1 --lr-encd 0.01 --lr-ally 0.001 --lr-advr 0.001 --alpha 0.5 --expt mnist --encd-ckpt ckpts/mnist/models/mnist_encd_pretrain_A.stop --ally-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_ally_0.stop --advr-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_advr_0.stop

# D: no
# python expt_mnist/eigan_training.py --device gpu --n-gpu 1 --n-ally 2 --n-advr 10 --batch-size 32 --n-epochs 100 --shuffle 0 --init-w 1 --lr-encd 0.1 --lr-ally 0.01 --lr-advr 0.01 --alpha 0.5 --expt mnist --encd-ckpt ckpts/mnist/models/mnist_encd_pretrain_A.stop --ally-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_ally_0.stop --advr-ckpts ckpts/mnist/models/mnist_encd_pretrain_A_advr_0.stop
