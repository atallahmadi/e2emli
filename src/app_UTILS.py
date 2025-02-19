import os
import gc
import cv2
import shutil
import numpy as np
import tensorflow as tf
import tifffile as tiff

from tqdm import tqdm
from skimage import io
from glob import glob
from PIL import Image
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model
from app_METRICS import weighted_dice_coef, weighted_IoU, get_lr_metric, weighted_categorical_crossentropy, dice_coef, IoU

# Constants and Config
Image.MAX_IMAGE_PIXELS=None
CLASSES_W = {'BR': 6.2715, 'BV': 7.5832, 'AVL': 0.7827, 'AVLW': 1.4901, 'BG': 0.3692}
RESOLUTIONS = ['2.5x', '5x', '10x', '20x']

# GPU Configuration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs available: {[device.name for device in physical_devices]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU available. Using CPU.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU

def pad_image(image, tile_size):
    h, w = image.shape[0], image.shape[1]
    h_pad, w_pad = (tile_size - h % tile_size) % tile_size, (tile_size - w % tile_size) % tile_size

    # Pad the image using replication of border pixels
    padded_image = cv2.copyMakeBorder(
        src=image, 
        top=0, 
        bottom=h_pad, 
        left=0, 
        right=w_pad, 
        borderType=cv2.BORDER_REPLICATE
    )
    return padded_image

def rescale_image(image, resolution, scale_down=True, size=(0, 0)):
    if resolution == "2.5x":
        scale_factor = 0.125 if scale_down else 8
    elif resolution == "5x":
        scale_factor = 0.25 if scale_down else 4
    elif resolution == "10x":
        scale_factor = 0.5 if scale_down else 2
    elif resolution == "20x":
        return image
    else:
        raise ValueError("Unsupported resolution")
    
    if len(image.shape) == 2:  # Grayscale image
        zoom_factors = (scale_factor, scale_factor)
    elif len(image.shape) == 3:  # RGB image
        zoom_factors = (scale_factor, scale_factor, 1)
    else:
        raise ValueError("Unsupported image dimensions")

    rescaled_image = zoom(image, zoom_factors, order=1 if scale_down else 3)
    if size != (0, 0):
        rescaled_image = rescaled_image[:size[1], :size[0]]

    return rescaled_image

def stitch_tiles(tile_paths, tile_size=256, colour=True):
    max_x, max_y = 0, 0
    tile_width, tile_height = tile_size, tile_size
    for tile_path in tile_paths:
        filename = os.path.basename(tile_path)
        parts = filename.split('_')
        x, y = int(parts[-2]), int(parts[-1].split(".")[0])
        max_x = max(max_x, x + tile_width)
        max_y = max(max_y, y + tile_height)
    stitched_image = np.zeros((max_y, max_x, 3), dtype=np.uint8) if colour else np.zeros((max_y, max_x), dtype=np.uint8)

    for tile_path in tile_paths:
        filename = os.path.basename(tile_path)
        parts = filename.split('_')
        x, y = int(parts[-2]), int(parts[-1].split(".")[0])
        tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)
        stitched_image[y:y+tile_height, x:x+tile_width] = tile

    return stitched_image

def _HistEQ(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def read_image(path, histeq=False):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = _HistEQ(x) if histeq else cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x / 255.0 
    x = x.astype(np.float32)
    
    return x

def read_stacked_imghm_mask(path, layer):
    p = os.path.normpath(path)
    parts = p.split(os.sep)
    tile_name = parts[-1]
    base, _ = p.split("RGB-input")
    tilepath = os.path.join(base.replace("FirstStage", "SecondStage"), 'WHM-input', '20x', tile_name.replace('RGB', f'{layer}_WHM'))

    img = read_image(path, histeq=True)
    hm = cv2.imread(tilepath, cv2.IMREAD_GRAYSCALE)
    hm = hm / 255
    hm = np.expand_dims(hm, axis=-1)
    hm = hm.astype(np.float32)
    img_hm = np.concatenate((img, hm), axis=-1)
    
    return img_hm

def extract_tiles(file_path, tile_size=256, overlap_percent=0):
    img_name = os.path.basename(file_path).split('.')[0]
    datasetpath = os.path.join(os.path.dirname(file_path), 'temp', 'FirstStage')
    os.makedirs(datasetpath, exist_ok=True)

    wsi = tiff.imread(file_path)[:, :, :3]
    wsi = wsi[:, :, ::-1]
    tile_step = tile_size if overlap_percent == 0 else int(tile_size * overlap_percent)
    for res in tqdm(RESOLUTIONS, desc="Resolutions"):
        os.makedirs(os.path.join(datasetpath, "RGB-input", res), exist_ok=True)
        img = pad_image(image=rescale_image(image=wsi, resolution=res, scale_down=True), tile_size=tile_size)
        tile_count = 0
        h, w = img.shape[0], img.shape[1]
        # Extract tiles from the padded image
        for x in range(0, w - tile_size + 1, tile_step):
            for y in range(0, h - tile_size + 1, tile_step):
                tile = img[y:y+tile_size, x:x+tile_size]
                tile_name = f"{img_name}_RGB_{res}_{x}_{y}.jpg"
                tile_path = os.path.join(datasetpath, "RGB-input", res, tile_name)
                tile_count += 1
                cv2.imwrite(tile_path, tile)

def run_eccn(file_path, model_folder):
    dataset_path = os.path.join(os.path.dirname(file_path), 'temp', 'FirstStage')
    for resolution in tqdm(RESOLUTIONS, desc=f"Running ECNN for resolution"):
        h5path = f"{model_folder}/ECNN_{resolution}.hdf5"
        custom_objects = {
            'DICE': weighted_dice_coef(np.ones((5,))),
            'IOU': weighted_IoU(np.ones((5,))),
            'lr': get_lr_metric(tf.keras.optimizers.Nadam(learning_rate=1e-3)),
            'loss': weighted_categorical_crossentropy(np.array(list(CLASSES_W.values())))
        }
        model = load_model(h5path, custom_objects=custom_objects)
        directories = sorted(glob(os.path.join(dataset_path, "RGB-input", resolution, "*")))
        for dir in tqdm(directories, desc=f"Processing directories"):
            p = dir.replace("RGB-input", "HM-input")
            sp = os.path.dirname(p)
            os.makedirs(sp, exist_ok=True)

            img = read_image(dir)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)
            prediction = prediction[0]

            for j in range(prediction.shape[-1]):
                name = os.path.basename(p).replace("RGB_", f"{list(CLASSES_W.keys())[j]}_HM_")
                out = f'{sp}/{name}'
                hm = (prediction[:, :, j] * 255).astype(np.uint8)
                hm = Image.fromarray(hm)
                hm.save(out)
                del hm
            del prediction, img
        del model
        gc.collect()
        tf.keras.backend.clear_session()

def reconstruct_heatmaps(file_path):
    resolutions = sorted(RESOLUTIONS, key=lambda x: float(x[:-1]), reverse=True)  # [20x, 10x, 5x, 2.5x]
    for layer in tqdm(CLASSES_W.keys(), desc=f"Reconstructing ECNN heatmaps"):
        imgs = []
        h, w = 0, 0
        for idx, res in enumerate(resolutions):
            all = sorted(glob(os.path.join(os.path.dirname(file_path), 'temp', "FirstStage", "HM-input", res, "*")))
            paths = [p for p in all if layer in p and f'_{res}_' in p]
            img = stitch_tiles(tile_paths=paths, colour=False)

            if idx == 0:  # 20x
                h, w = img.shape[0], img.shape[1]

            img = rescale_image(img, res, scale_down=False, size=(w, h))
            imgs.append(img)

            if idx == 3:  # 2.5x
                c1 = cv2.addWeighted(imgs[0], 0.5, imgs[1], 0.5, 0)  # hm20x + hm10x
                c2 = cv2.addWeighted(imgs[2], 0.5, imgs[3], 0.5, 0)  # hm5x + hm2.5x
                final_hm = cv2.addWeighted(c1, 0.5, c2, 0.5, 0)

                name = f'WHM_{layer}.jpg'
                p = os.path.join(os.path.dirname(file_path), 'temp', "FirstStage", "Reconstructed")
                os.makedirs(p, exist_ok=True)
                cv2.imwrite(os.path.join(p, name), final_hm)
            del img, paths

def extract_hm_tiles(file_path, tile_size=256, overlap_percent=0):
    res = "20x"   
    img_name = os.path.basename(file_path).split('.')[0]
    tile_step = tile_size if overlap_percent == 0 else int(tile_size * overlap_percent)
    datasetpath = os.path.join(os.path.dirname(file_path), 'temp', "SecondStage", "WHM-input", res)
    os.makedirs(datasetpath, exist_ok=True)
    
    for layer in CLASSES_W.keys():
        n = f'WHM_{layer}.jpg'
        p = os.path.join(os.path.dirname(file_path), "temp", "FirstStage", "Reconstructed", n)
        wHM = io.imread(p, True)
        tile_count = 0
        h, w = wHM.shape[0], wHM.shape[1]
        for x in range(0, w - tile_size + 1, tile_step):
            for y in range(0, h - tile_size + 1, tile_step):
                tile = wHM[y:y+tile_size, x:x+tile_size]
                tile_name = f"{img_name}_{layer}_WHM_20x_{x}_{y}.jpg"
                tile_path = os.path.join(datasetpath, tile_name)
                tile_count += 1
                cv2.imwrite(tile_path, tile)

def run_accn(file_path, model_folder):
    res = "20x"
    dataset_path = os.path.join(os.path.dirname(file_path), "temp", 'FirstStage')
    for layer in tqdm(CLASSES_W.keys(), desc=f"Running ACNN"):
        h5path = f"{model_folder}/ACNN_{layer}.hdf5"
        custom_objects = {
            'ACC': tf.keras.metrics.BinaryAccuracy(name='ACC'),
            'dice_coef': dice_coef,
            'IoU': IoU,
            'lr': get_lr_metric(tf.keras.optimizers.Nadam(learning_rate=1e-3)),
            'loss': 'binary_crossentropy'
        }

        model = load_model(h5path, custom_objects=custom_objects)

        directories = sorted(glob(os.path.join(dataset_path, "RGB-input", res, "*")))
        for dir in tqdm(directories, desc=f"Processing directories"):
            p = dir.replace("FirstStage", "SecondStage").replace("RGB-input", "HM-output")
            sp = os.path.dirname(p)
            os.makedirs(sp, exist_ok=True)

            img = read_stacked_imghm_mask(dir, layer)
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)
            prediction = prediction[0]

            # Threshold the prediction to create a binary image
            _, binary_prediction = cv2.threshold(prediction, 0.5, 1, cv2.THRESH_BINARY)

            name = os.path.basename(p).replace("RGB_", f"{layer}_").replace('jpg', 'png')
            out = f'{sp}/{name}'

            # Save the binary image
            cv2.imwrite(out, binary_prediction, [cv2.IMWRITE_PNG_BILEVEL, 1])
            del prediction, img
        del model
        gc.collect()
        tf.keras.backend.clear_session()

def reconstruct_final_heatmaps(file_path):
    res = "20x"
    wsi = tiff.imread(file_path)[:, :, 0]
    h, w = wsi.shape[0], wsi.shape[1]
    del wsi
    wsi_name = os.path.splitext(os.path.basename(file_path))[0]

    all = sorted(glob(os.path.join(os.path.dirname(file_path), "temp", "SecondStage", "HM-output", res, "*")))
    for layer in tqdm(CLASSES_W.keys(), desc=f"Reconstructing final heatmaps"):
        paths = [p for p in all if layer in p]
        img = stitch_tiles(tile_paths=paths, colour=False)
        img = img[:h, :w]

        o = os.path.join(os.path.dirname(file_path), wsi_name + "_masks")
        os.makedirs(o, exist_ok=True)
        img = Image.fromarray(img).convert('1')
        img.save(os.path.join(o, f'{layer}.tif'), compression='group3')
        del img

def clean_temp(file_path):
    temp = os.path.join(os.path.dirname(file_path), "temp")
    
    if os.path.exists(temp) and os.path.isdir(temp):
        shutil.rmtree(temp)
    print("Semantic Segmentation masks are created")

