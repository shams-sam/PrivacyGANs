download = False
data_folder = '../../data'
ckpt_folder = '../ckpts'

num_trains = {
    'mnist': 60000,
    'cifar_100': 50000,
    'celeba': 162770,
}

num_tests = {
    'mnist': 10000,
    'cifar_100': 10000,
    'celeba': 19867,
}

input_sizes = {
    'mnist': (28, 28),
    'cifar_100': (3, 32, 32),
    'celeba': (3, 32, 32) #(3, 218, 178),
}

num_channels = {
    'mnist': 1,
    'cifar_100': 3,
    'celeba': 3,
}
