"""
Datasets that the probing pipeline supports end-to-end.

For each `dataset_name` listed in `DATASET_CHOICES` we require:
  1. an attribute named `dataset_name` in `scripts.probing.prompts`
     (the list of prompts used to generate images), and
  2. an entry `dataset_name -> (text_groups, label_names)` in `LABEL_CONFIG`
     (used by CLIP to assign a binary label to every generated image).
"""

DATASET_CHOICES = [
    # gender (woman / man)
    "woman_man_prompts_neutral_v1",
    "woman_man_prompts_specific_v1",
    "woman_man_prompts_neutral_v2",

    # age (young / old)
    "age_prompts_specific_v2",
    "direct_prompts_adult_old_prompts_neutral_v1",
    "direct_prompts_adult_young_prompts_neutral_v1",

    # race (white vs other)
    "white_black_race_prompts_specific_v2",
    "white_asian_race_prompts_specific_v2",
    "white_indian_race_prompts_specific_v2",
    "direct_prompts_white_black_prompts_neutral_v1",
    "direct_prompts_white_asian_prompts_neutral_v1",
    "direct_prompts_white_indian_prompts_neutral_v1",

    # misc concept datasets used in the paper
    "usa_president_image_prompts",
    "disney_character_prompts",
    "man_face_prompts",
    "vincent_van_gogh_prompts",
    "material_prompts",
]

# Format: dataset_name -> (text_groups, label_names)
# `text_groups` is either a list of two strings (one prompt per class) or a
# list of two lists of strings (one group of prompts per class, CLIP picks
# the best match within each group).
LABEL_CONFIG = {
    # gender
    "woman_man_prompts_neutral_v1":  (["A photo of a woman", "A photo of a man"], ["woman", "man"]),
    "woman_man_prompts_specific_v1": (["A photo of a woman", "A photo of a man"], ["woman", "man"]),
    "woman_man_prompts_neutral_v2":  (["A photo of a woman", "A photo of a man"], ["woman", "man"]),

    # age
    "age_prompts_specific_v2":                       (["A photo of a young person", "A photo of an old person"],   ["young", "old"]),
    "direct_prompts_adult_old_prompts_neutral_v1":   ([["A photo of an adult person", "A photo of a young person"], ["A photo of an old person"]],   ["adult", "old"]),
    "direct_prompts_adult_young_prompts_neutral_v1": ([["A photo of an adult person", "A photo of an old person"],  ["A photo of a young person"]], ["adult", "young"]),

    # race
    "white_black_race_prompts_specific_v2":  (["A photo of a white person", "A photo of a black person"],  ["white", "black"]),
    "white_asian_race_prompts_specific_v2":  (["A photo of a white person", "A photo of an asian person"], ["white", "asian"]),
    "white_indian_race_prompts_specific_v2": (["A photo of a white person", "A photo of an indian person"], ["white", "indian"]),
    "direct_prompts_white_black_prompts_neutral_v1":  ([["A photo of a white person", "A photo of an indian person", "A photo of an asian person"], ["A photo of a black person"]], ["white", "black"]),
    "direct_prompts_white_asian_prompts_neutral_v1":  ([["A photo of a white person", "A photo of an indian person", "A photo of a black person"],  ["A photo of an asian person"]], ["white", "asian"]),
    "direct_prompts_white_indian_prompts_neutral_v1": ([["A photo of a white person", "A photo of a black person", "A photo of an asian person"],   ["A photo of an indian person"]], ["white", "indian"]),

    # misc
    "usa_president_image_prompts": ([["A photo of Joe Biden", "A photo of Barack Obama", "A photo of George W. Bush"], ["A photo of Donald Trump"]], ["other", "donald_trump"]),
    "disney_character_prompts":    ([["A photo of a human", "A photo of an animal"],                                   ["A photo of a mickie mouse"]], ["other", "mickie_mouse"]),
    "man_face_prompts":            (["A photo of a man with a beard", "A photo of a man without a beard"],             ["man_with_beard", "man_without_beard"]),
    "vincent_van_gogh_prompts":    (["A photo of a generic oil painting", "A photo of a painting by Vincent van Gogh"], ["generic_oil_painting", "vincent_van_gogh"]),
    "material_prompts":            ([["A photo of a metal object", "A photo of a wooden object", "A photo of a plastic object"], ["A photo of a glass object"]], ["other", "glass"]),
}


def label_config(name):
    """Return (text_groups, label_names) for `name`."""
    return LABEL_CONFIG[name]
