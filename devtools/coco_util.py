from collections import defaultdict
import json


def load_coco_dataset(path):
    """
    TODO
    """
    with open(path, "r") as f:
        dataset = json.load(f)
    return dataset


def filter_dataset(
    dataset, desired_cats=None, min_cats=0, max_cats=None, min_supercats=None,
    min_anns=0, max_anns=None
):
    """
    Args:
        dataset (dict): COCO dataset.
        desired_cats (List[int|str]): List of category names or ids that each
            returned image must contain. If omitted, images are not filtered
            by specific categories present.
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


if __name__ == "__main__":
    ds = load_coco_dataset("annotations/instances_val2017.json")
    ds = filter_dataset(ds, min_cats=6, max_anns=10, min_supercats=6)
    urls = [image["coco_url"] for image in ds["images"]]
