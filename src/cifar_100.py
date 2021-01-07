import config as cfg
from sklearn import preprocessing
from torchvision import datasets


class CIFAR100:
    _dataset = datasets.CIFAR100(cfg.data_folder, download=cfg.download)
    _coarse_to_fine = {
        'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
        'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
        'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
        'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        'large_carnivores':	['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large_man_made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'medium_sized_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        'non_insect_invertebrates':	['crab', 'lobster', 'snail', 'spider', 'worm'],
        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
        'reptiles':	['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    }

    _label_encoder = preprocessing.LabelEncoder()
    _label_encoder.fit(list(_coarse_to_fine.keys()))

    _idx_to_fine_class = {val: key for key,
                          val in _dataset.class_to_idx.items()}

    _fine_to_coarse = {}
    for coarse, fine_list in _coarse_to_fine.items():
        for fine in fine_list:
            _fine_to_coarse[fine] = coarse

    @classmethod
    def get_coarse_class_ids(cls, fine_class_ids):
        coarse_classes = [cls._fine_to_coarse[cls._idx_to_fine_class[int(_)]]
                          for _ in fine_class_ids]
        return cls._label_encoder.transform(coarse_classes)
