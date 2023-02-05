
"""

Env

"""
# BASE_PATH = "/Users/felixgerschner/git/ai-prototype/" # mac
BASE_PATH = "/home/fgerschner/git/ai-prototype/" # machine

ITERATIONS = 5

SOURCE_DATASETS = [
    # ("mechanicalseals_fulllight", ".png", 2, (4000, 3000)), # TEST
    # ("imagenet", None, None, None),
<<<<<<< HEAD
    ("dagm", ".PNG", 10, (512, 512)),
    # ("caltech101", ".jpg", 102, (300, 297)), # will be made after safe set is trained
=======
    # ("dagm", ".PNG", 10, (512, 512)),
    ("caltech101", ".jpg", 102, (300, 297)), # will be made after safe set is trained
>>>>>>> 831ad6715f31adea6aff5050013906b2dab03d8a
    # ("miniimagenet", ".JPEG", 100, (500, 375)), # crashed kernel TODO: test on workstation
]

TARGET_DATASETS = [
    ("mechanicalseals_fulllight", ".png", 2, (4000, 3000)),
]

TF_MODELS = [
    "vgg16", # suggested in paper
    # "vgg19",
    # "resnet50",
    # "resnet50V2",
    "resnet101", # suggested in paper
    # "resnet101V2",
    # "mobilenet", # suggested in paper
    # "mobilenetV2",
    # "mobilenetV3Large", 
    # "mobilenetV3Small", 
    "densenet121", # suggested in paper
]

TF_CLASSIFIER = [
    "custom"
]



K = [
    50,
    45,
    40,
    35,
    30,
    25,
    20,
    15,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1
]