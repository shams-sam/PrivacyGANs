mnist () {
    python pretrain.py --expt mnist --device gpu --gpu-id 1 \
	   --train-nets gan clf \
	   --optimizer sgd --resnet-layers 34 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 --load-w 0 \
	   --ckpt-g none\
	   --ckpt-d none\
	   --ckpt-clfs none \
	   --n-epochs 51 --lr-g 0.001 --lr-d 0.0001\
	   --lr-clfs 0.1 0.01 0.1 --weight-decays 1e-4 1e-4 1e-4 \
	   --milestones 1 25 40 50 --gamma 0.2
}

cifar () {
    python pretrain.py --expt cifar_100 --device gpu --gpu-id 0 \
	   --train-nets gan clf \
	   --optimizer sgd --num-layers 101 --n-classes 100 20 \
	   --batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/cifar_100/models/pretrain_g.stop \
	   --ckpt-d ../ckpts/cifar_100/models/pretrain_d.stop \
	   --ckpt-clfs ../ckpts/cifar_100/models/pretrain_clf_0.stop \
	   ../ckpts/cifar_100/models/pretrain_clf_1.stop\
	   --n-epochs 60 --lr-g 0.0001 --lr-d 0.0001 \
	   --lr-clfs 0.02 0.02 --weight-decays 1e-4 1e-4 5e-4 \
	   --milestones  60 120 --gamma 0.2
}

cifar_0 () {
    python cifar_train.py --epoch 160 --batch-size 256 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 100 --clf-num 0 
}

cifar_1 () {
    python cifar_train.py --epoch 160 --batch-size 256 --lr 0.1 --momentum 0.9 --wd 1e-4 -ct 100 --clf-num 1
}


celeba () {
    python pretrain.py --expt celeba --device gpu --gpu-id 0 1 2 \
	   --train-nets gan clf \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 2 2 2 2 2 2 2 2 \
	   2 2 2 2 2 2 2 2 2 2 \
	   2 2 2 2 2 2 2 2 2 2 \
	   2 2 2 2 2 2 2 2 2 2 \
	   --img-size 64 --batch-size 256 --test-batch-size 128 --subset 0.3 --init-w 1 --load-w 0 \
	   --ckpt-g none \
	   --ckpt-d none \
	   --ckpt-clfs none \
	   --n-epochs 21 --lr-g 0.000001 --lr-d 0.00001 --lr-clfs \
	   0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 \
	   0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 \
	   0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 \
	   0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 \
	   --weight-decays 1e-6 1e-6 5e-4 \
	   --milestones 20 40 50 --gamma 0.1
}


$1
