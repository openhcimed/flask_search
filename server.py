import sys
sys.path.append('../..')
import io
import json
import base64
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import torch
import torch.nn.functional as F
from skimage.filters import frangi, hessian
from deep_lesion.models.vae import LesionVAE
from deep_lesion.models.contrastive import LesionSimCLR


app = Flask(__name__)
CORS(app)


def get_lesion_size(bbox):
    coordinates = [float(i) for i in bbox.split(',')]
    x = abs(coordinates[2] - coordinates[0])
    y = abs(coordinates[0] - coordinates[1])

    return f'{x:.2f}, {y:.2f}'

def get_diameter(diameter):
    x, y = [float(i) for i in diameter.split(',')]

    return f'{x:.2f}, {y:.2f}'

# Read candidates and metadata
SKIP = [660, 6679, 6680, 9943, 11078, 11338, 18692, 19957, 22403]   # Skip non-accessible images
dl_info = pd.read_csv('data/DL_info.csv') # Metadata of lesions
dl_info = dl_info[~dl_info.index.isin(SKIP)]
dl_info = dl_info[dl_info.Train_Val_Test == 3].reset_index(drop=True)
dl_info['label'] = dl_info.Coarse_lesion_type - 1
dl_info['lesion_size'] = dl_info.Bounding_boxes.apply(get_lesion_size)
dl_info['diameter'] = dl_info.Lesion_diameters_Pixel_.apply(get_diameter)

raw_image = np.load('weights/test_df.npy')#[:, :, :, 0]  # Candidates, i.e. DeepLesion test
raw_image = raw_image.reshape(4927, 64, 64)
# images
emb_image = np.load('weights/emb_image.npy')  # Pre-computed embeddings of candidates
label = np.load('weights/label.npy')          # Label

# Load model weights
# vae = LesionVAE.load_from_checkpoint('./weights/vae.ckpt')
simclr = LesionSimCLR.load_from_checkpoint('weights/simclr.ckpt', gpus=1, num_samples=1, batch_size=1, dataset='lesion')
# vae.eval()
simclr.eval()

label_map = {0: 'Bone', 1: 'Abdomen', 2: 'Mediastinum', 3: 'Liver', 4: 'Lung', 5: 'Kidney', 6: 'Soft tissue', 7: 'Pelvis'}

def get_query_image(image_array, num_row, num_col):
    '''Convert 1D image array into 2D image'''
    image = []
    i = 0
    for r in range(num_col):
        row = []
        for c in range(num_row):
            row.append(image_array[i])
            i += 1
        image.append(row)
    image = np.array(image)
    return image / 255

def get_image_embedding(img):
    '''Get image embedding with VAE and SimCLR'''
    img = torch.tensor(img)
    img = F.interpolate(img.unsqueeze(dim=0).unsqueeze(dim=0), (224, 224)).float()
    with torch.no_grad():
        # emb_vae = vae.get_latent_vector(img).numpy()
        emb_simclr = simclr.get_latent_vector(img).numpy()

    return emb_simclr

def encode_img(img):
    img = Image.fromarray(np.uint8(img * 255) , 'L')
    data = io.BytesIO()
    img.save(data, 'JPEG')
    enc = base64.b64encode(data.getvalue())
    # enc = base64.b64encode(img.tobytes())

    return enc

def prepare_for_render(img):
    decoded_img = encode_img(img).decode('utf-8')

    return f'data:image/jpeg;base64,{decoded_img}'


#query = frangi(query, sigmas=(2, 3), scale_step=1, beta=5, gamma=0.00000002, black_ridges=True)
#frangi(emb_image, scale_range=(2, 3), scale_step=1, beta=5,
                                            # gamma=0.00000001, black_ridges=True))

def search(query_emb, k=8):
    '''Search the closest k neighbors of query image among all candidates'''
    distance = np.subtract(query_emb, frangi(emb_image, sigmas=1, scale_step=1, beta=6,
                                             gamma=0.00000002, black_ridges=True))
    distance = np.linalg.norm(distance, axis=1, ord=2)
    indices = np.argsort(distance)
    candidates = [
        (prepare_for_render(raw_image[i]),
         f'{distance[i]:.2f}',
         label_map[dl_info['label'][i]],
         dl_info['lesion_size'][i],
         dl_info['diameter'][i],
        ) for i in indices[:k]]

    return candidates

@app.route('/api/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        image_base64 = data['image']
        num_row = data['row']
        num_col = data['col']

        # Query
        image_array = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_array, dtype=np.uint8)
        query = get_query_image(image_array, num_row, num_col)
        # query = torch.tensor(query)
        # query = F.interpolate(query.unsqueeze(dim=0).unsqueeze(dim=0), (224, 224)).float().squeeze().squeeze()
        # query = query.numpy()
        tempquery = query
        query = frangi(query, sigmas=1, scale_step=1, beta=6,
                                             gamma=0.00000002, black_ridges=True)
        query_emb_simclr = get_image_embedding(query)

        # Search
        # candidates_vae = search(query_emb_vae)
        candidates_simclr = search(query_emb_simclr)

        return render_template(
            'index.html',
            query_path=prepare_for_render(tempquery),
            candidates_simclr=candidates_simclr,
            startx=data['startx'],
            starty=data['starty'],
            endx=data['endx'],
            endy=data['endy'],
            height=data['endy'] - data['starty'],
            width=data['endx']-data['startx'])
            # candidates_vae=candidates_vae
    else:
        return render_template('index.html')

@app.route('/', methods=['GET'])
def roi():
    return render_template('main/roi.html')


if __name__=="__main__":
    app.run("0.0.0.0", port=44443, debug=True)


# def get_annotation(annotations):
#     annotation = annotations[0]
#     for key, annotation_types in annotation['annotations'].items():
#         if annotation_types:
#             for row in annotation_types:
#                 left = row['handles']['textBox']['boundingBox']['left'] - row['handles']['textBox']["x"]
#                 json_data = {
#                     "width":row['handles']['textBox']['boundingBox']['width'],
#                     "height":row['handles']['textBox']['boundingBox']['height'],
#                     "left":left,
#                     "right": left+row['handles']['textBox']['boundingBox']['width'],
#                     "top":row['handles']['textBox']['boundingBox']['top'],
#                     "bottom":row['handles']['textBox']['boundingBox']['top']+row['handles']['textBox']['boundingBox']['height'],
#                     "uuid":row["uuid"],
#                     "x":row['handles']['textBox']["x"],
#                     "y":row['handles']['textBox']["y"],
#                     "length":row['length'] if row.get("length") else 0,
#                     "unit":row['unit'],
#                 }
#     return json_data