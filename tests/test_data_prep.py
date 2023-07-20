import unittest
import pandas as pd
from src.datadream.DD import DataPrep
import numpy as np

class TestDataPrep(unittest.TestCase):

    def setUp(self):
        self.texts = ['This one is in English', 'Celui ci est en franglais', 'Dieser ist auf Deutsch', 'Este es en espanol',
             'Questo e in italiano', 'これは日本語です', '这是中文']
        self.df = pd.DataFrame({'id': range(0, len(self.texts)*2, 2), 'text': self.texts, 'randomIntValue': [8]*len(self.texts), 'randomTxtValue': ['dsd']*len(self.texts)}).set_index('id')
        self.existing_map = pd.DataFrame(np.random.rand(len(self.texts), 2), index=self.df.index)
        self.existing_embeddings = pd.DataFrame(np.random.rand(len(self.texts), 768), index=self.df.index)
        #prep = DataPrep(self.df)
        #df, self.embeddings, self.map = prep.run(_translate=True)

    def test_run_without_translate(self):
        prep = DataPrep(self.df)
        df, embeddings, map = prep.run(embeddings=self.existing_embeddings, map=self.existing_map, translate=False)
        # Since we have no translation the shape of the dataframe should be the same
        self.assertTrue(self.df.shape[1]==df.shape[1])
        self.assertEqual(self.texts, df['text'].tolist())

    def test_run_with_translate(self):
        prep = DataPrep(self.df)
        df, embeddings, map = prep.run(embeddings=self.existing_embeddings, map=self.existing_map, translate=True)
        # After the translation we have more columns in the dataframe
        self.assertTrue(self.df.shape[1] < df.shape[1])
        # After the translation the texts are different
        self.assertNotEqual(self.texts, df['text'].tolist())

    def test_run_same_embeddings(self):
        prep = DataPrep(self.df)
        df, embeddings, map = prep.run(embeddings=self.existing_embeddings.copy(), map=self.existing_map, translate=False)
        # The embeddings returned should be the same as the ones passed
        self.assertTrue(self.existing_embeddings.equals(embeddings))
        # The index should be the same
        self.assertTrue(self.existing_embeddings.index.equals(embeddings.index))

    def test_run_same_map(self):
        prep = DataPrep(self.df)
        df, embeddings, map = prep.run(embeddings=self.existing_embeddings, map=self.existing_map.copy(), translate=False)
        # The embeddings returned should be the same as the ones passed
        self.assertTrue(self.existing_map.equals(map))
        # The index should be the same
        self.assertTrue(self.existing_map.index.equals(map.index))


