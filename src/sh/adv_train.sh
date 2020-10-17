mnist () {
    python adv_train.py --expt mnist --device gpu --gpu-id 1 \
	   --optimizer sgd --resnet-layers 34 --n-classes 10 2 2 \
	   --batch-size 256 --init-w 0 --load-w 1\
	   --ckpt-g ../ckpts/mnist/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/mnist/models/pretrain_clf_0.stop \
	   ../ckpts/mnist/models/pretrain_clf_1.stop \
	   ../ckpts/mnist/models/pretrain_clf_2.stop \
	   --n-epochs 51 --lr-g 0.001 \
	   --lr-clfs 0.1 0.1 0.1 --ei-array -0.33 0.33 -0.33 --weight-decays 1e-4 1e-4 \
	   --milestones 2 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}

cifar () {
    python adv_train.py --expt cifar_100 --device gpu --gpu-id 0 \
	   --optimizer sgd --resnet-layers 152 --n-classes 100 20 \
	   --batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/cifar_100/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/cifar_100/models/pretrain_clf_0.stop \
	   ../ckpts/cifar_100/models/pretrain_clf_1.stop \
	   --n-epochs 200 --lr-g 0.000001 \
	   --lr-clfs 0.1 0.1 --ei-array -0.5 0.5 --weight-decays 1e-4 1e-4 \
	   --milestones 2 60 120 160  --gamma 0.2
}



celeba_2 () {
    python adv_train.py --expt celeba --device gpu --gpu-id 0 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 0 0 0 0 0 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/celeba/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/celeba/models/pretrain_clf_0.stop \
	   ../ckpts/celeba/models/pretrain_clf_20.stop \
	   --n-epochs 51 --lr-g 1e-6 --lr-clfs \
	   1e-6 1e-6 --ei-array 0.5 -0.5\
	   --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}

celeba_3 () {
    python adv_train.py --expt celeba --device gpu --gpu-id 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 0 0 0 0 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/celeba/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/celeba/models/pretrain_clf_0.stop \
	   ../ckpts/celeba/models/pretrain_clf_1.stop \
	   ../ckpts/celeba/models/pretrain_clf_20.stop \
	   --n-epochs 51 --lr-g 1e-6 --lr-clfs \
	   1e-6 1e-6 1e-6 --ei-array 0.25 0.25 -0.5\
	   --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}


celeba_4 () {
    python adv_train.py --expt celeba --device gpu --gpu-id 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 0 2 0 0 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/celeba/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/celeba/models/pretrain_clf_0.stop \
	   ../ckpts/celeba/models/pretrain_clf_1.stop \
	   ../ckpts/celeba/models/pretrain_clf_3.stop \
	   ../ckpts/celeba/models/pretrain_clf_20.stop \
	   --n-epochs 51 --lr-g 1e-6 --lr-clfs \
	   1e-6 1e-6 1e-6 1e-6 --ei-array 0.167 0.167 0.167 -0.5\
	   --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}

celeba_5 () {
    python adv_train.py --expt celeba --device gpu --gpu-id 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 0 2 0 2 0 0 0 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/celeba/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/celeba/models/pretrain_clf_0.stop \
	   ../ckpts/celeba/models/pretrain_clf_1.stop \
	   ../ckpts/celeba/models/pretrain_clf_3.stop \
	   ../ckpts/celeba/models/pretrain_clf_5.stop \
	   ../ckpts/celeba/models/pretrain_clf_20.stop \
	   --n-epochs 51 --lr-g 1e-6 --lr-clfs \
	   1e-6 1e-6 1e-6 1e-6 1e-6 --ei-array 0.125 0.125 0.125 0.125 -0.5\
	   --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}

celeba_10 () {
    python adv_train.py --expt celeba --device gpu --gpu-id 0 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 2 2 2 2 2 2 2 0 \
	   0 0 0 0 0 0 0 0 0 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/celeba/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/celeba/models/pretrain_clf_0.stop \
	   ../ckpts/celeba/models/pretrain_clf_1.stop \
	   ../ckpts/celeba/models/pretrain_clf_2.stop \
	   ../ckpts/celeba/models/pretrain_clf_3.stop \
	   ../ckpts/celeba/models/pretrain_clf_4.stop \
	   ../ckpts/celeba/models/pretrain_clf_5.stop \
	   ../ckpts/celeba/models/pretrain_clf_6.stop \
	   ../ckpts/celeba/models/pretrain_clf_7.stop \
	   ../ckpts/celeba/models/pretrain_clf_8.stop \
	   ../ckpts/celeba/models/pretrain_clf_20.stop \
	   --n-epochs 51 --lr-g 1e-6 --lr-clfs \
	   1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 \
	   --ei-array 0.056 0.056 0.056 0.056 0.056 0.056 0.056 0.056 0.056 -0.5\
	   --weight-decays 1e-4 1e-4 \
	   --milestones 20 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}


celeba_20 () {
    python adv_train.py --expt celeba --device gpu --gpu-id 0 1 2 \
	   --arch resnet --optimizer sgd --num-layers 18 --n-classes \
	   2 2 2 2 2 2 2 2 2 2 \
	   2 2 2 2 2 2 2 2 2 0 \
	   2 0 0 0 0 0 0 0 0 0 \
	   --img-size 64 --subset 0.3 \
	   --batch-size 256 --test-batch-size 256 --init-w 0 --load-w 1 \
	   --ckpt-g ../ckpts/celeba/models/pretrain_g.stop \
	   --ckpt-clfs ../ckpts/celeba/models/pretrain_clf_0.stop \
	   ../ckpts/celeba/models/pretrain_clf_1.stop \
	   ../ckpts/celeba/models/pretrain_clf_2.stop \
	   ../ckpts/celeba/models/pretrain_clf_3.stop \
	   ../ckpts/celeba/models/pretrain_clf_4.stop \
	   ../ckpts/celeba/models/pretrain_clf_5.stop \
	   ../ckpts/celeba/models/pretrain_clf_6.stop \
	   ../ckpts/celeba/models/pretrain_clf_7.stop \
	   ../ckpts/celeba/models/pretrain_clf_8.stop \
	   ../ckpts/celeba/models/pretrain_clf_9.stop \
	   ../ckpts/celeba/models/pretrain_clf_10.stop \
	   ../ckpts/celeba/models/pretrain_clf_11.stop \
	   ../ckpts/celeba/models/pretrain_clf_12.stop \
	   ../ckpts/celeba/models/pretrain_clf_13.stop \
	   ../ckpts/celeba/models/pretrain_clf_14.stop \
	   ../ckpts/celeba/models/pretrain_clf_15.stop \
	   ../ckpts/celeba/models/pretrain_clf_16.stop \
	   ../ckpts/celeba/models/pretrain_clf_17.stop \
	   ../ckpts/celeba/models/pretrain_clf_18.stop \
	   ../ckpts/celeba/models/pretrain_clf_20.stop \
	   --n-epochs 51 --lr-g 1e-6 --lr-clfs \
	   1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 \
	   1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 1e-6 \
	   --ei-array 0.026 0.026 0.026 0.026 0.026 0.026 0.026 0.026 0.026 0.026\
	   0.026 0.026 0.026 0.026 0.026 0.026 0.026 0.026 0.026 -0.5\
	   --weight-decays 1e-4 1e-4 \
	   --milestones 10 20 40 70 90 --save-ckpts 10 20 30 40 50 --gamma 0.1
}

$1
