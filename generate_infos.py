import os
import pickle
import random

import tqdm
from PIL import Image

from utils import apply_ocr


def generate_infos(data_root, split, resize_factor=4):
    """Generates the training and val infos

    This function helps caches the compute-intensive OCR preprocessing required
    for the solution. 

    Args:
        data_root: the root folder where the dataset belongs
        split: the size of the train split, always in the range of [0, 1]
        resize_factor: the factor to resize the image by for the OCR, default value
            is 4.
    """

    train_infos = {
        "images": [],
        "words": [],
        "bboxes": [],
        "labels": [],
    }
    val_infos = {
        "images": [],
        "words": [],
        "bboxes": [],
        "labels": [],
    }

    for label in os.listdir(data_root):
        print(f"Generating infos for class {label}")
        image_paths = os.listdir(os.path.join(data_root, label))
        random.shuffle(image_paths)
        for i, example in tqdm.tqdm(enumerate(image_paths), total=len(image_paths)):
            img_path = os.path.join(data_root, label, example)

            image = Image.open(img_path).convert("RGB")

            # improves OCR performance
            if resize_factor != 1:
                image = image.copy().resize(
                    (image.size[0] * resize_factor, image.size[1] * resize_factor),
                    Image.LANCZOS,
                )
            words, bboxes = apply_ocr(image, return_boxes=True)

            # stratified split.
            if i <= split * len(image_paths):
                infos = train_infos
            else:
                infos = val_infos

            infos["images"].append(f"{label}/{example}")
            infos["words"].append(words)
            infos["bboxes"].append(bboxes)
            infos["labels"].append(label)

    return train_infos, val_infos



if __name__ == "__main__":
    train_infos, val_infos = generate_infos(
        data_root="./data/rvl_cdip_1000_samples",
        split=0.8,
    )

    with open("./data/train_infos.pkl", "wb") as pkl:
        pickle.dump(train_infos, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./data/val_infos.pkl", "wb") as pkl:
        pickle.dump(val_infos, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Train and validation infos generated")
    print("Size of train set", len(train_infos["images"]))
    print("Size of val set", len(val_infos["images"]))
