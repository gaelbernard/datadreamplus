import pandas as pd
from src.datadream.DD import DataDream, DataPrep
import requests
requests.packages.urllib3.util.connection.HAS_IPV6 = False
import sqlite3


db = sqlite3.connect('/Users/gaeberna/Library/Mobile Documents/com~apple~CloudDocs/Dev/2023-05-08-ActiveLearningAzure/data/wiki/documents', check_same_thread=False)

#perc = pd.read_sql_query(f"select indexid, documents.id, data from documents INNER JOIN sections ON sections.id = documents.id", self.hugDB).set_index(['id', 'indexid'])['data'].astype('str').str.replace('{"percentile": ','',regex=False).str.replace('}','',regex=False).astype(float)
d = pd.read_sql_query(f"select indexid, documents.id, data, text from documents INNER JOIN sections ON sections.id = documents.id", db)
d['percentile'] = d['data'].astype('str').str.replace('{"percentile": ', '', regex=False).str.replace('}','',regex=False).astype(float)
d['text'] = d['id'].astype('str') + '. ' + d['text'].astype('str')
d.set_index('id', inplace=True)
d.drop(columns=['indexid', 'data'], inplace=True)



prep = DataPrep(d)
df, embeddings, map = prep.run()

cdd = DataDream('wikipedia_complete')
cdd.upload(df, embeddings, map, 'Entire Wikipedia', 'Txt downloaded from Txtai', allow_overwrite=True, encrypt=False)


print (df.head().to_string())
print (embeddings.head().to_string())
print (map.head().to_string())
exit()
