# python expt_titanic/autoencoder_training.py --device gpu --n-ally 1 --n-advr 1 --dim 588 --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 1 --lr 0.0001 --expt titanic

# python expt_mimic/autoencoder_training.py --device gpu --n-ally 1 --n-advr 1 --dim 176 --test-size 0.1 --batch-size 1024 --n-epochs 1001 --shuffle 1 --lr 0.0001 --expt mimic

python expt_mnist/autoencoder_training.py --device gpu --n-ally 1 --n-advr 1 --dim 332 --test-size 0.3 --batch-size 1024 --n-epochs 1001 --shuffle 1 --lr 0.0001 --expt mnist