import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


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
                (y + h) / image_height
            ]
            bboxes.append(bbox)

        assert len(words) == len(
            bboxes
        ), "Not as many words as there are bounding boxes"

        return words, bboxes

    return words
