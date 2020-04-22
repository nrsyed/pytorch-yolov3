from collections import defaultdict
import json


def load_coco_dataset(path):
    """
    TODO
    """
    with open(path, "r") as f:
        dataset = json.load(f)
    return dataset


def images_containing_categories(dataset, categories):
    """
    TODO
    """
    if all(isinstance(cat, int) for cat in categories):
        cat_ids = set(categories)
    elif all(isinstance(cat, str) for cat in categories):
        # Mapping of category names to category ids.
        cat_name_to_id = {
            cat["name"]: cat["id"] for cat in dataset["categories"]
        }
        cat_ids = set(cat_name_to_id[cat] for cat in categories)

    image_id_to_cats = defaultdict(set)
    for ann in dataset["annotations"]:
        image_id_to_cats[ann["image_id"]].add(ann["category_id"])

    image_id_to_image = {image["id"]: image for image in dataset["images"]}

    images = []
    for image_id, cats in image_id_to_cats.items():
        if cat_ids.intersection(cats) == cat_ids:
            images.append(image_id_to_image[image_id])
    return images


if __name__ == "__main__":
    ds = load_coco_dataset("annotations/instances_val2017.json")
    images = images_containing_categories(ds, ["dog", "person", "bicycle"])
    urls = [image["coco_url"] for image in images]
