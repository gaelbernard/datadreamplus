# Datadream

## Prerequisites

### Encryptions & Keys
At the root of your project, you should have the following two files:


- `config.py` containing the encryption keys:
```python
ENCRYPTION_KEY = 'eeb...{SECRET_KEY}...'
LIBRE_TRANSLATE_API_KEY = '118..{SECRET_KEY}...'
```
- `credentials.json` containing the encryption keys for gcloud (downloaded from the gcloud console):
```json
{
  "type": "service_account",
  "project_id": "lithe-style-232907",
  "private_key_id": "...{SECRET_KEY}...",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMII..{SECRET_KEY}...",
  "client_email": "661978949901-compute@developer.gserviceaccount.com",
  "client_id": "109522766032154966558",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/661978949901-compute%40developer.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}


It is very important to install
pip install googletrans==4.0.0-rc1
Otherwise, the translation won't work
```

### Install the package
```bash 
pip install https://github.com/gaelbernard/datadreamplus/raw/main/dist/datadream-0.0.1-py3-none-any.whl --force-reinstall --no-cache-dir
```


## Getting started

### Import the library
```python
from datadream.DD import DataDream, DataPrep
```

### List all datasets
To list all datasets available in the cloud, use the following:
```python
CloudDualData.list_dataset(verbose=True)
```

### Load a dataset
All datasets are composed of three dataframes that can be loaded independently. Each dataset will be saved locally in the `data` folder. When you run the command again, it compares the MD5 and only download the files that have changed.
```python
# Load the dataset helloworld
d = DataDream('helloworld')

data = d.load_data()
embeddings = d.load_embeddings()
map = d.load_map()
```

### Upload a dataset
Let's say we have the following dataset
```python
import pandas as pd
df = pd.DataFrame({'text': ['I am in English', 'Me too'], 'other': [0, 1], 'color':['red','blue']}).set_index('other')
print (df)
``` 
```bash
                  text color
other                       
0      I am in English   red
1               Me too  blue
```

#### Data Preparation
DataPrep takes such a dataset and produce the three datasets required by DataDream. If needed, it also translate the text to the target language (default to False).
The only requirement is to have a column "text" that will be used to embed the text.
The first time you run this code, it can be quite slow as it needs to download BERT and UMAP models.
```python
prep = DataPrep(df)
df, embeddings, map = prep.run()
```

In case you already computed the embeddings, you can use the following so it won't recompute them.
In this case, the embeddings should be a dataframe with the same index as the original dataframe.
```python
prep = DataPrep(df, embeddings=embeddings)
df, embeddings, map = prep.run()
```

Same goes for the map...
```python
prep = DataPrep(df, embeddings=embeddings, map=map)
df, embeddings, map = prep.run()
```

You can also translate the text. In this case, the translated text will be stored in the column 'text' and the original text in an additional column
```python
prep = DataPrep(df, translate=True)
df, embeddings, map = prep.run()
```

#### Upload
Uploading to the cloud is as simple as:
```python
cdd = DataDream('helloworld')
cdd.upload(df, embeddings, map, 'top 5% of all wikipedia pages (in terms of viewcount)', 'dataset downloaded from txtai')
```

The upload will not work if a dataset already exists with the same name. You can force the upload by setting `allow_overwrite=True` in the upload function.

### Build the wheel
```bash
 cd dist
 pip wheel ../. 

```