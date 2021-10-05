import numpy as np


class_names = np.array(
    [
        "__background__",
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "010_potted_meat_can",
        "011_banana",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "024_bowl",
        "025_mug",
        "035_power_drill",
        "036_wood_block",
        "037_scissors",
        "040_large_marker",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick",
    ]
)
class_names.setflags(write=False)

class_names_symmetric = np.array(
    [
        "024_bowl",
        "036_wood_block",
        "051_large_clamp",
        "052_extra_large_clamp",
        "061_foam_brick",
    ]
)
class_names_symmetric.setflags(write=False)
class_ids_symmetric = np.array(
    [np.where(class_names == name)[0][0] for name in class_names_symmetric],
    dtype=np.int32,
)
class_ids_symmetric.setflags(write=False)

class_names_asymmetric = class_names[
    ~np.isin(class_names, class_names_symmetric)
    & ~(class_names == "__background__")
]
class_names_asymmetric.setflags(write=False)
class_ids_asymmetric = np.array(
    [np.where(class_names == name)[0][0] for name in class_names_asymmetric],
    dtype=np.int32,
)
class_ids_asymmetric.setflags(write=False)
