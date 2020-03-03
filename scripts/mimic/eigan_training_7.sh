
#A
python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 32768 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.01 --lr-ally 0.0001 --lr-advr 0.0001 --alpha 0.5 --expt mimic --num-allies 7
python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 32768 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr 0.0001 --alpha 0.5 --expt mimic --num-allies 7
python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 32768 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.0001 --lr-advr 0.0001 --alpha 0.5 --expt mimic --num-allies 7
python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 32768 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.00001 --lr-ally 0.0001 --lr-advr 0.0001 --alpha 0.5 --expt mimic --num-allies 7

#B
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 2048 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.01 --lr-ally 0.00001 --lr-advr-1 0.00001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 2048 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.00001 --lr-advr-1 0.00001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 2048 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.00001 --lr-advr-1 0.00001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic

#C
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.0001 --lr-advr-1 0.000001 --lr-advr-2 0.000001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.00001 --lr-advr-1 0.000001 --lr-advr-2 0.000001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.000001 --lr-advr-1 0.000001 --lr-advr-2 0.000001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic

#D
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.000001 --lr-advr-1 0.000001 --lr-advr-2 0.000001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 2048 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.000001 --lr-advr-1 0.000001 --lr-advr-2 0.000001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 2048 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.00001 --lr-advr-1 0.00001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic

#E
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.00001 --lr-advr-1 0.00001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 2048 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.00001 --lr-advr-1 0.00001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 2048 --hidden-dim 4096 --leaky 0 --activation tanh --test-size 0.3 --batch-size 512 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.0001 --lr-ally 0.00001 --lr-advr-1 0.00001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic

#D: first
# python expt_titanic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --n-advr-1 1 --n-advr-2 3 --dim 1400 --hidden-dim 2800 --leaky 0 --activation tanh --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.00001 --lr-ally 0.00001 --lr-advr-1 0.000001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt titanic
# python expt_titanic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --n-advr-1 1 --n-advr-2 3 --dim 1400 --hidden-dim 2800 --leaky 0 --activation tanh --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.00001 --lr-ally 0.00001 --lr-advr-1 0.000001 --lr-advr-2 0.000001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt titanic
# python expt_titanic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --n-advr-1 1 --n-advr-2 3 --dim 1400 --hidden-dim 2800 --leaky 0 --activation tanh --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.00001 --lr-ally 0.00001 --lr-advr-1 0.000001 --lr-advr-2 0.0000001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt titanic

#E
# python expt_titanic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --n-advr-1 1 --n-advr-2 3 --dim 1400 --hidden-dim 2800 --leaky 0 --activation tanh --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.00001 --lr-ally 0.00001 --lr-advr-1 0.000001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt titanic
# python expt_titanic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --n-advr-1 1 --n-advr-2 3 --dim 700 --hidden-dim 1400 --leaky 0 --activation tanh --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.00001 --lr-ally 0.00001 --lr-advr-1 0.000001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt titanic
# python expt_titanic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --n-advr-1 1 --n-advr-2 3 --dim 2800 --hidden-dim 5600 --leaky 0 --activation tanh --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.00001 --lr-ally 0.00001 --lr-advr-1 0.000001 --lr-advr-2 0.00001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt titanic

#F
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 4 --hidden-dim 8 --leaky 0 --activation tanh --test-size 0.3 --batch-size 16384 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 8 --hidden-dim 16 --leaky 0 --activation tanh --test-size 0.3 --batch-size 16384 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 16 --hidden-dim 32 --leaky 0 --activation tanh --test-size 0.3 --batch-size 16384 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 32 --hidden-dim 64 --leaky 0 --activation tanh --test-size 0.3 --batch-size 16384 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 64 --hidden-dim 128 --leaky 0 --activation tanh --test-size 0.3 --batch-size 16384 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 128 --hidden-dim 256 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 512 --hidden-dim 1024 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 1024 --hidden-dim 2048 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 2048 --hidden-dim 4096 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 4096 --hidden-dim 8192 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic

#G
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 64 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 128 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 256 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 512 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 2048 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 4096 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic
# python expt_mimic/eigan_training_n.py --device gpu --n-gpu 1 --n-ally 1 --dim 256 --hidden-dim 512 --leaky 0 --activation tanh --test-size 0.3 --batch-size 9192 --n-epochs 1001 --shuffle 0 --init-w 1 --lr-encd 0.001 --lr-ally 0.0001 --lr-advr-1 0.0001 --lr-advr-2 0.0001 --alpha 0.5 --g-reps 1 --d-reps 1 --expt mimic