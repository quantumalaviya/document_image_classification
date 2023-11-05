import pickle

from datasets import Dataset, Image

from utils import visualize_documents


def dataset_generator(infos_pkl_path, root_dir="data/rvl_cdip_1000_samples"):
    """Generates a HF dataset from infos.

    Args:
        infos_pkl_path: the path of the infos pkl file to load the dataset from
        root_dir: root data directory where the rvl-cdip images are stored.
    """
    with open(infos_pkl_path, "rb") as pkl:
        infos = pickle.load(pkl)

    # add the root data to the image path
    infos["images"] = list(map(lambda x: f"{root_dir}/{x}", infos["images"]))

    # renaming column names for downstream coding standards
    infos["image"] = infos.pop("images")
    infos["bbox"] = infos.pop("bboxes")

    ds = Dataset.from_dict(infos)
    ds = ds.cast_column("image", Image())
    return ds


if __name__ == "__main__":
    pkl_path = "data/val_infos.pkl"

    ds = dataset_generator(pkl_path)
    img, bboxes = ds[0]["image"].convert("RGB"), ds[0]["bbox"]

    visualize_documents(img, bboxes)
