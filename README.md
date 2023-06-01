# Front_UI install:
https://github.com/umb-deephealth/deephealth-annotate

## weights

Note: weights need to be loaded from https://www.dropbox.com/sh/nok6uvc892jy71y/AAChDBdm-9u4eJNsGiro0iWEa?dl=0 and put in weights/ folder

```
mkdir weights
cd weights
curl -L https://www.dropbox.com/sh/nok6uvc892jy71y/AAChDBdm-9u4eJNsGiro0iWEa?dl=1 -o weights.zip
unzip weights.zip
```
## conda env
You need miniconda (https://docs.conda.io/en/latest/miniconda.html)

install and re-login (python 3.7)
```
conda create -n your_env python=3.7
```

and activate it

```
conda activate your_env
```

and then install the pip packages

```
pip install -r requirements.txt
```

## To run:

```
cd ../
python server_vae.py
```
## Drag-drop and Crop an image and click the search button:
<img width="609" alt="drag-drop image" src="https://user-images.githubusercontent.com/22691548/174459159-dea3bca4-13cb-4ed0-afb4-0f5f87cd66d1.png">


