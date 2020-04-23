from collections import defaultdict
import colorsys
import json
import os

import numpy as np


def load_coco_dataset(path):
    with open(path, "r") as f:
        dataset = json.load(f)
    return dataset


def filter_dataset(
    dataset, desired_cats=None, min_cats=0, max_cats=None, min_supercats=None,
    min_anns=0, max_anns=None
):
    """
    Filter images of a dataset based on the object categories and/or number
    of annotations they contain.

    Args:
        dataset (dict): COCO dataset.
        desired_cats (List[int|str]): List of category names or ids that each
            returned image must contain.
        min_cats (int): Minimum number of distinct object categories (classes)
            in each returned image.
        max_cats (int): Maximum number of distinct object categories (classes)
            in each returned image. If None, all images with >min_cats
            represented classes are included.
        min_supercats (int): Minimum number of supercategories represented in
            each returned image; if None, any number of supercategories are
            allowed.
        min_anns (int): Minimum number of annotations (objects) present in
            each returned image.
        max_anns (int): Maximum number of annotations (objects) presented in
            each returned image; if None, all images with >min_anns
            annotations are included.

    Returns:
        Dict representing filtered dataset containing only images (and their
        annotations) matching the provided criteria.
    """
    # Mapping of category id to supercat name.
    cat_id_to_supercat = {
        cat["id"]: cat["supercategory"] for cat in dataset["categories"]
    }

    # Mapping of image id to annotations, image id to categories represented
    # in image, and image id to supercategories represented in an image.
    image_id_to_anns = defaultdict(list)
    image_id_to_cats = defaultdict(set)
    image_id_to_supercats = defaultdict(set)
    for ann in dataset["annotations"]:
        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        image_id_to_anns[image_id].append(ann)
        image_id_to_cats[image_id].add(cat_id)
        image_id_to_supercats[image_id].add(cat_id_to_supercat[cat_id])

    # If `desired_cats` supplied, convert category names to category ids
    # (if not already int) and create set of desired ids.
    desired_cat_ids = None
    if desired_cats is not None:
        if all(isinstance(cat, int) for cat in desired_cats):
            desired_cat_ids = set(desired_cats)
        elif all(isinstance(cat, str) for cat in desired_cats):
            # Mapping of category names to category ids.
            cat_name_to_id = {
                cat["name"]: cat["id"] for cat in dataset["categories"]
            }
            desired_cat_ids = set(cat_name_to_id[cat] for cat in desired_cats)

    # Filter images by desired criteria.
    image_ids = []
    for image in dataset["images"]:
        image_id = image["id"]
        num_anns = len(image_id_to_anns[image_id])
        num_cats = len(image_id_to_cats[image_id])
        num_supercats = len(image_id_to_supercats[image_id])

        image_cats = image_id_to_cats[image_id]

        if (
            (num_cats >= min_cats)
            and (not max_cats or num_cats <= max_cats)
            and (num_anns >= min_anns)
            and (not max_anns or num_anns <= max_anns)
            and (not desired_cat_ids or desired_cat_ids.issubset(image_cats))
            and (not min_supercats or num_supercats >= min_supercats)
        ):
            image_ids.append(image_id)

    filtered_dataset = {
        "info": dataset["info"],
        "licenses": dataset["licenses"],
        "images": [
            image for image in dataset["images"] if image["id"] in image_ids
        ],
        "annotations": [
            ann for ann in dataset["annotations"]
            if ann["image_id"] in image_ids
        ],
        "categories": dataset["categories"]
    }
    return filtered_dataset


def match_ids(dataset, reference_dataset):
    """
    Update category ids and image ids in `dataset` to match those
    in `reference_dataset` (based on category names and image filenames).
    """
    cat_name_to_old_id = dict()
    for cat in dataset["categories"]:
        cat_name_to_old_id[cat["name"]] = cat["id"]

    old_cat_id_to_new_id = dict()
    for cat in reference_dataset["categories"]:
        old_id = cat_name_to_old_id[cat["name"]]
        new_id = cat["id"]
        old_cat_id_to_new_id[old_id] = new_id

    image_fname_to_old_id = dict()
    for image in dataset["images"]:
        image_fname_to_old_id[image["file_name"]] = image["id"]

    old_image_id_to_new_id = dict()
    for image in reference_dataset["images"]:
        old_id = image_fname_to_old_id[image["file_name"]]
        new_id = image["id"]
        old_image_id_to_new_id[old_id] = new_id

    # Mapping of reference dataset image ids to images.
    ref_image_id_to_image = {
        image["id"]: image for image in reference_dataset["images"]
    }

    for image in dataset["images"]:
        new_id = old_image_id_to_new_id[image["id"]]
        image["id"] = new_id
        image["height"] = ref_image_id_to_image[new_id]["height"]
        image["width"] = ref_image_id_to_image[new_id]["width"]

    for ann in dataset["annotations"]:
        ann["category_id"] = old_cat_id_to_new_id[ann["category_id"]]
        ann["image_id"] = old_image_id_to_new_id[ann["image_id"]]

    dataset["categories"] = reference_dataset["categories"]


def unique_colors(num_colors):
    for H in np.linspace(0, 1, num_colors, endpoint=False):
        rgb = colorsys.hsv_to_rgb(H, 1.0, 1.0)
        bgr = (int(255 * rgb[2]), int(255 * rgb[1]), int(255 * rgb[0]))
        yield bgr


def draw_coco(dataset, image_dir):
    """
    Display images from a COCO dataset with bboxes superimposed on them.
    Similar to `inference.draw_boxes()`.

    Args:
        dataset (dict): Dict representing a COCO dataset.
        image_dir (str): Path to directory of images from the COCO dataset.
    """
    import cv2

    image_id_to_annotations = defaultdict(list)
    for ann in dataset["annotations"]:
        image_id_to_annotations[ann["image_id"]].append(ann)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in dataset["categories"]}
    cat_id_to_idx = {
        cat["id"]: i for i, cat in enumerate(dataset["categories"])
    }
    colors = list(unique_colors(len(dataset["categories"])))

    for image in sorted(dataset["images"], key=lambda x: x["id"]):
        img = cv2.imread(os.path.join(image_dir, image["file_name"]))

        annotations = image_id_to_annotations[image["id"]]
        for ann in annotations:
            tl_x, tl_y, w, h = [int(coord) for coord in ann["bbox"]]
            br_x = tl_x + w
            br_y = tl_y + h
            cat_id = ann["category_id"]
            cat_name = cat_id_to_name[cat_id]
            color = colors[cat_id_to_idx[cat_id]]

            cv2.rectangle(
                img, (tl_x, tl_y), (br_x, br_y), color=color, thickness=2
            )

            cv2.rectangle(
                img, (tl_x + 1, tl_y + 1),
                (tl_x + int(8 * len(cat_name)), tl_y + 18),
                color=(20, 20, 20), thickness=cv2.FILLED
            )
            cv2.putText(
                img, cat_name, (tl_x + 1, tl_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), thickness=1
            )

        cv2.imshow("bboxes", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    """
    dataset = load_coco_dataset("annotations/instances_val2017.json")
    dataset = filter_dataset(dataset, min_cats=6, max_anns=10, min_supercats=6)
    urls = [image["coco_url"] for image in dataset["images"]]

    dst_dir = os.path.abspath("../sample_dataset")
    dst_img_dir = os.path.join(dst_dir, "images")
    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir, exist_ok=True)

    for url in urls:
        req = requests.get(url)
        image_fname = os.path.split(url)[1]
        image_fpath = os.path.join(dst_img_dir, image_fname)

        if not os.path.exists(image_fpath):
            with open(image_fpath, "wb") as f:
                f.write(req.content)

    dataset_fpath = os.path.join(dst_dir, "sample.json")
    with open(dataset_fpath, "w") as f:
        json.dump(dataset, f, indent=2)
    """
    pass
