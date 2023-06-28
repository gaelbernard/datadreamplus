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
```

### Install the package
```bash 
pip install '/Users/gaeberna/Library/Mobile Documents/com~apple~CloudDocs/Dev/2023-06-27-datadreamplus/datadreamplus/dist/datadream-0.0.1-py3-none-any.whl' --force-reinstall
```


## Getting started

### List all datasets
To list all datasets available in the cloud, use the following:
```python
CloudDualData.list_dataset(verbose=True)
```

### Load a dataset
```python
CloudDualData.list_dataset(verbose=True)
```
