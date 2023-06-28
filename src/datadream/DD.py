import pandas as pd
from sentence_transformers import SentenceTransformer
from google.cloud import storage
import json
import hashlib
import pathlib
import cryptpandas as crp
from config import *
import sys
import shutil
import pickle
from googletrans import Translator
from langdetect import detect
import time
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'credentials.json'


class DataDream:
    BUCKET_NAME = 'ep-embedding'
    cloud = storage.Client()

    '''
    This class is used to upload and download the dataframes prepared with DataPrep.
    It is called CloudDualData because it contains two dataframes stored in the cloud:
    - df: the original dataframe with data and at least a column "text"
    - embeddings: the embedding of the column "text"
    Both dataframes are stored in the cloud and locally (will download it if not present locally)
    We can access them independently with the methods get_df() and get_embeddings()
    '''
    def __init__(self, blob_name, data_path='data'):
        '''
        Initialize the class with the name of the blob and the path where the data will be stored locally
        :param blob_name: name of the blob on google blob
        :param data_path: folder where the data will be stored locally
        '''

        self.blob_name = blob_name
        self.bucket = self.cloud.bucket(self.BUCKET_NAME)
        self.blobs = {
            'data': self.bucket.blob(f'data_{self.blob_name}'),
            'embeddings': self.bucket.blob(f'embeddings_{self.blob_name}'),
            'map': self.bucket.blob(f'map_{self.blob_name}'),
        }
        self.path = data_path
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def _load(self, type):
        '''
        Load the data from the cloud or locally
        :param type: 'data', 'embeddings'
        :return: a dataframe
        '''
        blob = self.blobs[type]

        assert type in self.blobs.keys()
        assert self.blob_name in DataDream.list_blobs_name()
        assert blob.exists()

        filename = f'{self.path}/{type}_{self.blob_name}.pkl'
        metadata = self.bucket.get_blob(f'data_{self.blob_name}').metadata

        # Before loading from the cloud, we check if the data already exists locally
        print (filename)
        if os.path.exists(filename) and metadata:
            if metadata[f'md5_{type}'] == hashlib.md5(pathlib.Path(filename).read_bytes()).hexdigest():
                print('LOADING: The local version is up to date!')
            else:
                print('LOADING: the local version is outdated, downloading from the cloud...')
                os.remove(filename)
                blob.download_to_filename(filename)
        else:
            print('LOADING: no local version found, downloading from the cloud...')
            blob.download_to_filename(filename)

        if metadata:
            if metadata['encrypted'] == 'True' and type=='data':
                decrypted_df = crp.read_encrypted(path=filename, password=ENCRYPTION_KEY)
                decrypted_df.to_pickle(filename)
                return decrypted_df

        return pd.read_pickle(filename)

    def load_data(self):
        return self._load('data')

    def load_embeddings(self):
        return self._load('embeddings')

    def load_map(self):
        return self._load('map')

    def upload(self, data, embeddings, map, description, source, allow_overwrite=False, encrypt=True):
        '''
        Upload a dataframe and the embedding to the cloud
        :param df: Dataframe containing the text that was embedded ('column')
        :param embeddings: Embedding in a df format (with same index than df)
        :return:
        '''
        assert data.index.duplicated().sum() == 0
        assert 'text' in data.columns
        assert data.index.equals(embeddings.index)
        assert map.index.equals(embeddings.index)
        if self.blob_name in DataDream.list_blobs_name() and not allow_overwrite:
            raise ValueError('The blob already exists, use allow_overwrite=True to overwrite it')
        for b in self.blobs.keys():
            assert not self.blobs[b].exists() or allow_overwrite

        queue = {
            'embeddings': embeddings,
            'map': map,
            'data': data,
        }
        md5 = {}
        for type, df in queue.items():

            print (f'uploading {type} ...')
            file = f'temp_{type}.pkl'

            # Extraction to temporary files (before uploading to the cloud)
            # Encryption: Extra layer of protection, we encrypt the data before uploading it
            # (can be skipped for non-confidential data)
            if encrypt and type=='data':
                crp.to_encrypted(data, password=ENCRYPTION_KEY, path=file)
            else:
                data.to_pickle(file)

            md5[type] = hashlib.md5(pathlib.Path(file).read_bytes()).hexdigest()

            # Upload to the cloud
            self.blobs[type].upload_from_filename(file)

            # Delete the temporary file
            os.remove(file)


        # We put the metadata in the data blob only
        # ...avoid to have two versions of the metadata
        sample_data = data.sample(max(10,data.shape[0]))

        # Upload the metadata
        metadata = {
            'nrecords': data.shape[0],
            'ncolumns': data.shape[1],
            'columns': data.columns.tolist(),
            'size_data_mo': sys.getsizeof(data)*1e-6,
            'size_embeddings_mo': sys.getsizeof(embeddings)*1e-6,
            'size_map_mo': sys.getsizeof(map)*1e-6,
            'description': description,
            'source': source,
            'sample_data': sample_data.to_json(orient='index'),
            'sample_embedding': embeddings.loc[sample_data.index,:].to_json(orient='index'),
            'sample_map': map.loc[sample_data.index,:].to_json(orient='index'),
            'md5_data': md5['data'],
            'md5_embeddings': md5['embeddings'],
            'md5_map': md5['map'],
            'encrypted': encrypt,
        }
        self.blobs['data'].metadata = metadata
        self.blobs['data'].patch()

    def delete_blob(self):
        '''
        Delete the blob from the cloud
        :return:
        '''
        p = 'Are you sure you want to delete the blob from the cloud? This cannot be undone (y/n): '
        if input(p) == 'y':
            for blob in self.blobs.values():
                if blob.exists():
                    blob.delete()
                    print ('Deleting', blob.name, '...')

    @staticmethod
    def list_blobs_name():
        '''
        List all the blobs name already taken on the cloud
        :return: a list of strings
        '''
        return [blob.name[5:] for blob in DataDream.cloud.list_blobs(DataDream.BUCKET_NAME) if blob.name.startswith('data_')]

    @staticmethod
    def list_dataset(verbose=True):
        '''
        List all the datasets available in the cloud
        :param verbose: Print the list of datasets and samples
        :return: metadata_df: a dataframe containing the metadata
                 dct_data: a dictionary containing the dataframes of data (key: blob name)
                 dct_embeddings: a dictionary containing the dataframes of embeddings (key: blob name)
        '''
        metadata_df = {}
        dct_data = {}
        dct_embeddings = {}

        # Loop over all the blobs
        for blob in DataDream.cloud.list_blobs(DataDream.BUCKET_NAME):

            # Since the metadata are stored in the data blob, we skip the embedding
            if not blob.name.startswith('data_'):
                continue

            name = blob.name[len('data_'):]
            metadata = {
                'size_mo': blob.size*0.000001,
                'updated': blob.updated,
                'md5_hash': blob.md5_hash,
                'language_model': DataPrep.MODEL,
            }
            if blob.metadata:
                metadata.update(blob.metadata)
                dct_data[name] = pd.DataFrame(json.loads(blob.metadata['sample_data']))
                dct_embeddings[name] = pd.DataFrame(json.loads(blob.metadata['sample_embedding']))
                del metadata['sample_data']
                del metadata['sample_embedding']

            metadata_df[name] = metadata

        # Build dataframes
        metadata_df = pd.DataFrame(metadata_df).transpose()

        if verbose:
            for i, r in metadata_df.iterrows():
                print ('##################')
                print (f'DATASET: {i}')
                print ()
                print (r.to_string())
                print ('## sample data:')
                print (dct_data[i].to_string())
                print ('## sample embeddings:')
                print (dct_embeddings[i])
                print ()
                print ()
        return metadata_df, dct_data, dct_embeddings

class DataPrep:
    '''
    Class to prepare the data for the embedding (before uploading it to the cloud)
    '''
    MODEL = 'all-mpnet-base-v2'

    def __init__(self, df, path='data'):
        '''
        We start from a dataframe.
        The only requirement is to have a column 'text'
        :param df:
        '''
        assert 'text' in df.columns
        assert df.index.duplicated().sum() == 0
        self.df = df
        self.df['text'] = df['text'].astype(str)
        self.translator = None
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def run(self, embeddings=None, map=None, translate=False, umap_cloud_path='umap_reducer_150_0_1000_squared.pkl'):
        '''
        Run the data preparation
        :param translate:u if True, we translate the text to english.
            We keep the original text in the df, but we put the translated version in the column text
        :param embeddings: Use embedding to represent the text.
            In case the text was already embedded, you can provide it as a dataframe so we avoid computational cost and we just check that the index matches
            Otherwise, just pass None, and it will compute the embeddings
        :param map: map the embeddings in a 2D space.
            If not provided we do it using UMAP
        :return: Two dataframes:
            - the original dataframe with the column 'text' translated to english if translate=True
            - the embeddings
            - the map
        '''
        if translate:
            self.translate()

        # If embeddings is not provided we compute it using the SentenceTransformer
        # otherwise we just check that the index matches
        if embeddings is None:
            embeddings = self._embed()
        else:
            assert isinstance(embeddings, pd.DataFrame)
            assert self.df.index.equals(embeddings.index)
            assert np.all(embeddings.dtypes == np.float64)
            assert embeddings.shape[1]>127

        # If map is not provided we compute it using UMAP
        # otherwise we just check that the index matches
        if map is None:
            map = pd.DataFrame(self._umap(embeddings, umap_cloud_path), index=embeddings.index)
        else:
            assert isinstance(map, pd.DataFrame)
            assert self.df.index.equals(map.index)


        return self.df, embeddings, map

    def _embed(self, batch_size=100):
        '''
        Embed the text using the SentenceTransformer.
        It will perform it in batches.
        In case of interruption, it will resume from where it stopped.

        :return: a dataframe with the embeddings
            we keep the same index as the original dataframe
        '''
        model = SentenceTransformer(self.MODEL)
        size = self.df.shape[0]

        # We save the file in a folder named md5
        hash = hashlib.md5(str(self.df.to_dict()).encode('utf-8')).hexdigest()
        folder = f'{self.path}/{hash}_embedding'

        # Create a temporary files
        if not os.path.exists(folder):
            os.makedirs(folder)

        # We check if we already have some embeddings (instead of starting from scratch)
        dfs = []
        start_index = self._batch_get_last_row_done(folder)

        # Do the loop by batch of 100
        for i in range(start_index, size, batch_size):
            ldf = self.df.iloc[i:i + batch_size, :]
            pd.DataFrame(model.encode(ldf['text'].values), index=ldf.index.tolist()).to_pickle(f'{folder}/{i}.pkl')

        # Merge into one dataframe
        embeddings = self._batch_merge_and_delete(folder)

        return embeddings

    def _umap(self, embeddings, umap_cloud_path, batch_size=10):
        local_path_umap = f'{self.path}/{umap_cloud_path}'
        # If umap is not found locally we download it from the cloud
        if not os.path.exists(local_path_umap):
            bucket = DataDream.cloud.bucket(DataDream.BUCKET_NAME)
            bucket.blob(umap_cloud_path).download_to_filename(local_path_umap)

        with open(local_path_umap, 'rb') as f:
            reducer = pickle.load(f)

        # Do the map by batches
        size = self.df.shape[0]

        # We save the file in a folder named md5
        hash = hashlib.md5(str(self.df.to_dict()).encode('utf-8')).hexdigest()
        folder = f'{self.path}/{hash}_map'

        # Create a temporary folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        # We check if we already have some map coordinates (instead of starting from scratch)
        start_index = self._batch_get_last_row_done(folder)

        # Do the loop by batch of 100
        for i in range(start_index, size, batch_size):
            lembeddings = embeddings.iloc[i:i + batch_size, :]
            map = reducer.transform(lembeddings)
            pd.DataFrame(map, index=lembeddings.index.tolist()).to_pickle(f'{folder}/{i}.pkl')

        # Merge into one dataframe
        map = self._batch_merge_and_delete(folder)
        return map

    def translate(self, batch_size=100):
        '''
        Translate the text to english
        :return: None
        '''

        '''
        translate the text in batches
        In case of interruption, it will resume from where it stopped.

        :return: a dataframe with the translations
            we keep the same index as the original dataframe
        '''
        size = self.df.shape[0]

        # We save the file in a folder named md5
        hash = hashlib.md5(str(self.df.to_dict()).encode('utf-8')).hexdigest()
        folder = f'{self.path}/{hash}_translation'

        # Create a temporary folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        # We check if we already have some translations (instead of starting from scratch)
        start_index = self._batch_get_last_row_done(folder)

        # Do the loop by batch of 100
        for i in range(start_index, size, batch_size):
            ldf = self.df.iloc[i:i + batch_size, :]
            translations = []
            for text in ldf['text'].tolist():
                translations.append(self._translate_text(text))
            pd.DataFrame(translations, index=ldf.index.tolist()).to_pickle(f'{folder}/{i}.pkl')

        # Merge into one dataframe
        translations = self._batch_merge_and_delete(folder)

        self.df['original_text'] = self.df['text']
        self.df['text'] = translations['translation']
        self.df = self.df.join(translations[['trad_detected_language', 'trad_status']])
        return translations

    def _translate_text(self, text, max_retries=5):
        '''
        Translate text from any language to english
        (Using libretranslate)
        :param text: text to translate
        :param max_retries: number of retries if translation fails
        :return: translated text
        '''
        assert isinstance(text, str)
        assert len(text)>0

        if self.translator is None:
            self.translator = Translator()

        # Detect language
        retry_count = 0
        delay = 1
        try:
            trad_detected_language = detect(text)
        except:
            print("language detection failed for text: " + text)
            return {'translation': text, 'trad_detected_language': 'fail', 'trad_status': 'language detection failed'}
        if trad_detected_language == 'en':
            return {'translation': text, 'trad_detected_language': 'en', 'trad_status': 'no translation needed'}

        while retry_count < max_retries:
            try:
                translated = self.translator.translate(text, src=trad_detected_language, dest='en')
                return {'translation': translated.text, 'trad_detected_language': trad_detected_language, 'trad_status': f'successfully translated (retry_count: {retry_count})'}
            except:
                if retry_count < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    retry_count += 1
                else:
                    return {'translation': text, 'trad_detected_language': trad_detected_language, 'trad_status': f'translation failed'}

    def _batch_get_last_row_done(self, folder):
        dfs = []
        start_index = 0
        for file in os.listdir(folder):
            if file.endswith(".pkl"):
                dfs.append(pd.read_pickle(f'{folder}/{file}'))
        if len(dfs) > 0:
            dfs = pd.concat(dfs)
            start_index = dfs.shape[0]
            print('found existing items, starting from index ', start_index)

        return start_index

    def _batch_merge_and_delete(self, folder):
        dfs = []
        for file in os.listdir(folder):
            if file.endswith(".pkl"):
                dfs.append(pd.read_pickle(f'{folder}/{file}'))
        dfs = pd.concat(dfs).loc[self.df.index, :]
        assert len(dfs) == self.df.shape[0]
        assert dfs.index.equals(self.df.index)
        shutil.rmtree(folder)

        return dfs


if __name__ == '__main__':
    t = ['abc']*99
    example = pd.DataFrame({'text': t, 'text2': 0.34*len(t), 'other': range(len(t)), 'other2': range(len(t))})
    example.set_index('other', inplace=True)

    prep = DataPrep(example)

    embedding = pd.DataFrame(np.random.rand(len(t), 768), index=example.index)

    # time the mapping
    start = time.time()
    df, embeddings, map = prep.run(embeddings=pd.DataFrame(embedding, index=example.index), translate=False)


    print (time.time() - start)
    exit()
    print (map)
    exit()
    cdd = DataDream('helloworld')
    #cdd.delete_blob()
    cdd.upload(example, embeddings, map, 'test set', 'synthetic data for testing')
    exit()


    #metadata_df, dct_data, dct_embeddings = CloudDualData.list_dataset(verbose=True)

    #exit()
    #
    # # load a dataset from the cloud
    #cdd = CloudDualData('wikipedia')
    #data = cdd.load_data()
    # embeddings = cdd.load_embeddings()
    # print (data)
    # exit()
    #embeddings = pd.read_pickle('embeddings.pkl')
    #data = pd.read_pickle('data.pkl')
    #print (data)
    #exit()
    cdd = DataDream('helloworld')
    embeddings = cdd.load_embeddings()
    data = cdd.load_data()
    print (embeddings)
    cdd.delete_blob()
    exit()
    print (data)

    cdd2 = DataDream('helloworld')
    prep = DataPrep(data)
    df, embeddings, map = prep.run(embeddings=embeddings, translate=False)
    print (map)

    cdd2.upload(data, embeddings, map, 'helloworld', 'helloworld', encrypt=False, allow_overwrite=True)
    exit()
    #cdd.upload(data, embeddings, 'Top 5% (in pageviews) of all wikipedia pages in English', 'Database from txtai downloaded here: https://huggingface.co/NeuML/txtai-wikipedia.', encrypt=False)

    # take df with text, use embedding, and upload it to the cloud
    t = ['This one is in English', 'Celui ci est en franglais', 'Dieser ist auf Deutsch', 'Este es en espanol', 'Questo e in italiano', 'これは日本語です', '这是中文', '이것은 한국어입니다', 'هذا باللغة العربية', 'Это на русском', 'यह हिंदी में है', 'এটি বাংলায়', 'ఇది తెలుగులో', 'ഇത് മലയാളത്തിൽ', 'این فارسی است', 'ეს ქართულია', 'Toto']
    example = pd.DataFrame({'text': t, 'other': range(len(t))})
    example.set_index('other', inplace=True)
    prep = DataPrep(example)
    df, embeddings, map = prep.run(translate=True)


    print ('second run')
    df, embeddings, map = prep.run(embeddings=embeddings, translate=False)

    print (embeddings)
    cdd = DataDream('helloworld')
    cdd.upload(df, embeddings, map, 'Wikipedia (EN): using lead paragraph', 'Model from HuggingFace', allow_overwrite=True, encrypt=False)
    exit()
    metadata_df, dct_data, dct_embeddings = DataDream.list_dataset(verbose=True)
    #print (data)
    # delete a dataset from the cloud
    #cdd.delete_blob()


