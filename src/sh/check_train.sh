mnist_resnet18 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnet34 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 34 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnet50 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 50 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnet101 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 101 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnet152 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 152  --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnet164 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --arch resnet --optimizer sgd --num-layers 164 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnext50 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --arch resnext --optimizer sgd --num-layers 50  --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

mnist_resnext101 () {
    python check_train.py --expt mnist --device gpu --gpu-id 1 \
	   --arch resnext --optimizer sgd --num-layers 101  --n-classes 10 2 2 \
	   --batch-size 256 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 51 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}


mnist_wideresnet101 () {
    python check_train.py --expt mnist --device gpu --gpu-id 0 \
	   --arch wideresnet --optimizer sgd --num-layers 101  --n-classes 10 2 2 \
	   --batch-size 1024 --init-w 1 \
	   --ckpt-g ../ckpts/mnist/models/adv_train_g.bkp \
	   --n-epochs 101 \
	   --lr-clfs 0.1 0.1 0.1 --weight-decays 5e-4 \
	   --milestones 2 40 70 90 --gamma 0.1
}

cifar () {
python check_train.py --expt cifar_100 --device gpu --gpu-id 0 --n-classes 100 20 \
       --batch-size 128 --init-w 1\
       --ckpt-g ../ckpts/cifar_100/models/adv_train_g.stop \
       --n-epochs 101 \
       --lr-clfs 0.1 0.1 --weight-decays 5e-4 \
       --milestones 2 20 40  --gamma 0.2
}

celeba_2 () {
    python check_train.py --expt celeba --device gpu --gpu-id 0 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 0 0 0 0 0 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 128 --init-w 1 \
	   --ckpt-g ../ckpts/celeba/models/adv_train_resnet18_ei_2_g.stop\
	   --n-epochs 51 \
	   --lr-clfs 1e-1 1e-1 --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 50 --gamma 0.1
}


celeba_3 () {
    python check_train.py --expt celeba --device gpu --gpu-id 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 0 0 0 0 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 128 --init-w 1 \
	   --ckpt-g ../ckpts/celeba/models/adv_train_resnet18_ei_3_g.stop\
	   --n-epochs 51 \
	   --lr-clfs 1e-1 1e-1 1e-1 --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 50 --gamma 0.1
}

celeba_4 () {
    python check_train.py --expt celeba --device gpu --gpu-id 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 0 2 0 0 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 128 --init-w 1 \
	   --ckpt-g ../ckpts/celeba/models/adv_train_resnet18_ei_4_g.stop\
	   --n-epochs 51 \
	   --lr-clfs 1e-1 1e-1 1e-1 1e-1 --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 50 --gamma 0.1
}

celeba_5 () {
    python check_train.py --expt celeba --device gpu --gpu-id 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 0 2 0 2 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 128 --init-w 1 \
	   --ckpt-g ../ckpts/celeba/models/adv_train_resnet18_ei_5_g.stop\
	   --n-epochs 51 \
	   --lr-clfs 1e-1 1e-1 1e-1 1e-1 1e-1 --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 50 --gamma 0.1
}


celeba_10 () {
    python check_train.py --expt celeba --device gpu --gpu-id 0 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 2 2 2 2 2 2 2 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 128 --init-w 1 \
	   --ckpt-g ../ckpts/celeba/models/adv_train_resnet18_ei_10_g_20.stop\
	   --n-epochs 51 --lr-clfs \
	   1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 \
	   --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 50 --gamma 0.1
}

$1
