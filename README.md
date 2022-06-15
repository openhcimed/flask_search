# flask_search

## weights

Note: weights need to be loaded from https://www.dropbox.com/sh/nok6uvc892jy71y/AAChDBdm-9u4eJNsGiro0iWEa?dl=0 and put in weights/ folder

```
mkdir weights
cd weights
curl -L https://www.dropbox.com/sh/nok6uvc892jy71y/AAChDBdm-9u4eJNsGiro0iWEa?dl=1 -o weights.zip
unzip weights.zip
```
## conda env
we need miniconda (https://docs.conda.io/en/latest/miniconda.html)

install and re-login

then create the env from .yml file

```
conda env create -f environment.yml
```

and activate it

```
conda activate kristin
```

and then install the pip packages

```
pip install -r requirements.txt
```

To run:

```
cd ../
python server.py
```
