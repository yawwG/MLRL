import numpy as np
import skimage.transform
from cv2.ximgproc import guidedFilter
from PIL import Image, ImageDraw, ImageFont

def save_image(name, image_np, output_path=""):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.jpg".format(name))
def save_image2(originalimage, name, image_np, output_path=""):
    p = np_to_pil2(originalimage,image_np)
    p.save(output_path + "{}.jpg".format(name))
def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def np_to_pil2(input, img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    # ar = np.uint8(img_np[0])
    if img_np.max()>1:
        img = img_np
        ar = np.uint8(img)
    else:
        img = skimage.img_as_float(img_np[0])  # 先转换成uint16的格式
        img = (img - img.min()) / (img.max() - img.min())
        ar = np.uint8(img * 255)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        if img_np.shape[0] == 3:
            ar = ar.transpose(1, 2, 0)

    input = np.clip(input * 255, 0, 255).astype(np.uint8)
    input = input.transpose(1, 2, 0)

    merged = Image.new("RGB", (512, 512), (0, 0, 0))
    mask = Image.new("L", (512, 512), (10))
    input = Image.fromarray(input)
    ar = Image.fromarray(ar)
    merged.paste(input, (0, 0))
    merged.paste(ar, (0, 0), mask)

    return merged

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    # img = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    if img_np.max()>1:
        img = img_np
    else:
        img=(img_np / 2 + 0.5)*255
    ar = np.uint8(img)
    # ar = np.uint8(img * 255)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        # assert img_np.shape[0] == 3, img_np.shape
        if img_np.shape[0] == 3:
            ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def t_matting(original_image,mask_out_np):
    refine_t = guidedFilter(original_image.transpose(1, 2, 0).astype(np.float32),
                            mask_out_np[0].astype(np.float32), 50, 1e-4)

    return np.array([np.clip(refine_t, 0.1, 1)])

def normalize(similarities, method="norm"):

    if method == "norm":
        return (similarities - similarities.mean(axis=0)) / (similarities.std(axis=0))
    elif method == "standardize":
        return (similarities - similarities.min(axis=0)) / (
            similarities.max(axis=0) - similarities.min(axis=0)
        )
    else:
        raise Exception("normalizing method not implemented")
