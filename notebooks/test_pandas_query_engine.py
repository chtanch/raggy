# %%
import phoenix as px
px.launch_app().view()

import pandas as pd

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.experimental.query_engine.pandas import PandasQueryEngine

llm = Ollama(model="phi3:mini-4k",temperature=0.0)

#%%
df = pd.read_excel("./data/job_descriptions.xlsx")
query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)

#%%
response = query_engine.query("A candidate has expertise in AWS. From the job descrpitions, what are the positions that are suitable for him?")
print(f'response: {response}')