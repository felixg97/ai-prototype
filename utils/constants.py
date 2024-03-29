
"""

Env

"""
# BASE_PATH = "/Users/felixgerschner/git/ai-prototype/"  # mac
BASE_PATH = "/media/wic/projects/felix-gerschner/git/ai-prototype/" # machine
# BASE_PATH = "/home/fgerschner/git/ai-prototype/" # machine
# BASE_PATH = "/media/wic/projects1/felix-gerschner/git/ai-prototype/" # low machine

ITERATIONS = 5

SOURCE_DATASETS = [
    ("imagenet", None, None, None),
    ("dagm", ".PNG", 10, (512, 512)),
    ("caltech101", ".jpg", 102, (300, 297)),
    # ("miniimagenet", ".JPEG", 100, (500, 375)), # crashed kernel TODO: test on workstation
]

TARGET_DATASETS = [
    ("mechanicalseals_fulllight", ".png", 2, (4000, 3000)),
]

TF_MODELS = [
    "vgg16",  # suggested in paper
    # "vgg19",
    # "resnet50",
    # "resnet50V2",
    "resnet101",  # suggested in paper
    # "resnet101V2",
    # "mobilenet", # suggested in paper
    # "mobilenetV2",
    # "mobilenetV3Large",
    # "mobilenetV3Small",
    # "densenet121", # suggested in paper
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
