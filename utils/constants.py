
"""

Env

"""

BASE_PATH = "/Users/felixgerschner/git/ai-prototype/"

ITERATIONS = 5


SOURCE_DATASETS = [
    # ("imagenet", None, None),
    # ("dagm", ".PNG", 10, (512, 512)),
    # ("caltech101", ".jpg", 101, (300, 297)), # will be made after safe set is trained
    # ("miniimagenet", ".JPEG", 100, (500, 375)), # crashed kernel TODO: test on workstation
]

TARGET_DATASETS = [
    ("mechanicalseals_fulllight", ".png", 2, (4000, 3000)),
]

TF_MODELS = [
    "VGG16", # suggested in paper
    # "VGG19",
    # "ResNet50",
    # "ResNet50V2",
    "ResNet101", # suggested in paper
    # "ResNet101V2",
    # "MobileNet", # suggested in paper
    # "MobileNetV2", 
    # "MobileNetV3Large", 
    # "MobileNetV3Small", 
    "DenseNet121", # suggested in paper
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