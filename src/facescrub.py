import os

import config as cfg
import pandas as pd
from torchvision import datasets


class FaceScrub:
    _dataset = datasets.ImageFolder(os.path.join(
        cfg.data_folder, 'FaceScrub/split/train'))
    _idx_to_class = {val: key for key, val in _dataset.class_to_idx.items()}
    _df = pd.read_csv(os.path.join(
        cfg.data_folder, 'FaceScrub/labels/id2gender.csv'), index_col=False)
    print(_df.head())
    _id_2_gender = {}
    for id, g in zip(_df.id.tolist(), _df.gender.tolist()):
        _id_2_gender[id.replace(' ', '_')] = 0 if g == 'male' else 1

    @classmethod
    def get_gender(cls, target_ids):
        names = [cls._idx_to_class[int(_)] for _ in target_ids]
        gender = [cls._id_2_gender[_] for _ in names]

        return gender
