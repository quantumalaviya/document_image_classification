import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image


def visualize_documents(image, bboxes=[], save=False, show=True, filename="test.jpg"):
    """visualization util for document images

    This basic utility function can either save or display document pictures
    with the option of displaying boxes over the words.

    Args:
      image: A PIL image on which to perform the OCR.
      bboxes: list of the bboxes to be plotted. The supported format is xyxy
          i.e. (left, top, width, height). Defaults to `[]`.
      save: whether to save the generated image or not, defaults to False
      show: whether to save the generated image or now, defaults to True
      filename: if save=True, this is the location that will be sent to
          cv2.imwrite, defaults to "test.jpg".
    """
    for box in bboxes:
        assert len(box) == 4, "wrong bounding box input"

    image_width, image_height = image.size
    image = np.array(image).copy()

    for box in bboxes:
        image = cv2.rectangle(
            image,
            (
                int(box[0] * image_width),
                int(box[1] * image_height),
            ),
            (
                int(box[2] * image_width),
                int(box[3] * image_height),
            ),
            color=[255, 0, 0],
            thickness=1,
        )

    if show:
        plt.imshow(image)
        plt.show()

    if save:
        # TODO: exception handling
        cv2.imwrite(filename, image[..., ::-1])


def apply_ocr(image, return_boxes=False):
    """apply_ocr applies ocr on PIL images using tesseract

    This function returns the words as detected on an image and uses the tesseract
    engine. Optionally, it can return bounding boxes for the words as well to
    compensate for the lack of visual information when OCR is performed.

    Since the scope of this project is limited to tesseract, a function is used
    as opposed to a class for this.

    Args:
      image: A PIL image on which to perform the OCR.
      return_boxes: whether to return boxes with the words or not, defaults
          to `False`.

    Image input: any PIL image that is supported by pytesseract APIs
    Output: One of:
        -> (words, bboxes): if return_boxes=True
        -> (words): if return_boxes=False
    the bboxes are normalized to the range of (0, 1)
    """

    image_width, image_height = image.size
    data = pytesseract.image_to_data(image, output_type="dict")
    words, left, top, width, height = (
        data["text"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
    )

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]

    if return_boxes:
        # extract bounding box information
        left = [
            coord for idx, coord in enumerate(left) if idx not in irrelevant_indices
        ]
        top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
        width = [
            coord for idx, coord in enumerate(width) if idx not in irrelevant_indices
        ]
        height = [
            coord for idx, coord in enumerate(height) if idx not in irrelevant_indices
        ]

        # turn coordinates into (left, top, left+width, top+height) format
        bboxes = []
        for x, y, w, h in zip(left, top, width, height):
            bbox = [
                x / image_width,
                y / image_height,
                (x + w) / image_width,
                (y + h) / image_height,
            ]
            bboxes.append(bbox)

        assert len(words) == len(
            bboxes
        ), "Not as many words as there are bounding boxes"

        return words, bboxes

    return words


def normalize_box(box):
    """Helper function to normalize boxes on a scale of 0-1000"""
    return [
        int(1000 * box[0]),
        int(1000 * box[1]),
        int(1000 * box[2]),
        int(1000 * box[3]),
    ]


def encode_example(example, tokenizer, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
    """Encodes an example to prepare it for the model forward pass

    Args:
      example: a single example dict containing the images, words, labels and
          optionally the boxes.
      tokenizer: the huggingface tokenizer to be used for the words. Optionally,
          can be any tokenizer that extends the HF Tokenizer class.
      max_seq_length: the max sequence length after which inputs will be clipped
          defaults to 512.
      pad_token_box: The values to pad the bboxes with, defaults to [0, 0, 0, 0].
    """
    del example["image"]
    words = example["words"]
    del example["words"]
    bboxes = example["bbox"]
    normalized_word_boxes = list(map(normalize_box, bboxes))

    assert len(words) == len(normalized_word_boxes)

    token_boxes = []
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))

    # Truncation of token_boxes
    special_tokens_count = 2
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    encoding = tokenizer(" ".join(words), padding="max_length", truncation=True)
    # Padding of token_boxes up the bounding boxes to the sequence length.
    input_ids = tokenizer(" ".join(words), truncation=True)["input_ids"]
    padding_length = max_seq_length - len(input_ids)
    token_boxes += [pad_token_box] * padding_length
    encoding["bbox"] = token_boxes
    encoding["labels"] = example["labels"]

    assert len(encoding["input_ids"]) == max_seq_length
    assert len(encoding["attention_mask"]) == max_seq_length
    assert len(encoding["token_type_ids"]) == max_seq_length
    assert len(encoding["bbox"]) == max_seq_length

    return encoding


def encode_example_v2(example, processor):
    # take a batch of images
    images = example["image"].convert("RGB")
    del example["image"]
    encoded_inputs = processor(
        images,
        text = example["words"],
        boxes = example["bbox"],
        padding="max_length",
        truncation=True
    )

    # add labels
    encoded_inputs["labels"] = example["labels"]

    return encoded_inputs
