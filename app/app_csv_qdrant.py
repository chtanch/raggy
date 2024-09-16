import streamlit as st
import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Initialize Qdrant Client
qdrant_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_qdrant_collection():
    # Create a Qdrant collection if it doesn't already exist
    qdrant_client.recreate_collection(
        collection_name="csv_data_vectors",
        vectors_config=VectorParams(size=384, distance="Cosine"),
    )


def upload_data_to_qdrant(df):
    # Convert dataframe rows to vectors and upload to Qdrant
    vectors = model.encode(
        df.astype(str).values.tolist()
    )  # Convert each row into a string and then embed
    points = [
        PointStruct(id=i, vector=vectors[i], payload={"row_data": row.to_dict()})
        for i, row in df.iterrows()
    ]
    qdrant_client.upsert(collection_name="csv_data_vectors", points=points)


def search_qdrant(query):
    # Search for the closest vectors to the query in Qdrant
    query_vector = model.encode([query])[0]  # Encode the query as a vector
    result = qdrant_client.search(
        collection_name="csv_data_vectors",
        query_vector=query_vector,
        limit=5,  # You can adjust this
    )
    return result


def main():
    st.title("Chat with your CSV")

    uploaded_file = st.file_uploader("**Upload a CSV file**", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("**Data preview:**")
        st.write(df.head())

        create_qdrant_collection()
        upload_data_to_qdrant(df)

        instruction_str = (
            "1. Convert the query to executable Python code using Pandas.\n"
            "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
            "3. The code should represent a solution to the query.\n"
            "4. PRINT ONLY THE EXPRESSION.\n"
            "5. Do not quote the expression.\n"
        )

        pandas_prompt_str = (
            "You are working with a pandas dataframe in Python.\n"
            "The name of the dataframe is `df`.\n"
            "This is the result of `print(df.head())`:\n"
            "{df_str}\n\n"
            "Follow these instructions:\n"
            "{instruction_str}\n"
            "Query: {query_str}\n\n"
            "Expression:"
        )

        response_synthesis_prompt_str = (
            "Given an input question, synthesize a response from the query results.\n"
            "Query: {query_str}\n\n"
            "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
            "Pandas Output: {pandas_output}\n\n"
            "Response: "
        )

        pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
            instruction_str=instruction_str, df_str=df.head(5)
        )
        pandas_output_parser = PandasInstructionParser(df)
        response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
        llm = Ollama(model="phi3")

        qp = QP(
            modules={
                "input": InputComponent(),
                "pandas_prompt": pandas_prompt,
                "llm1": llm,
                "pandas_output_parser": pandas_output_parser,
                "response_synthesis_prompt": response_synthesis_prompt,
                "llm2": llm,
            },
            verbose=True,
        )
        qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
        qp.add_links(
            [
                Link("input", "response_synthesis_prompt", dest_key="query_str"),
                Link(
                    "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
                ),
                Link(
                    "pandas_output_parser",
                    "response_synthesis_prompt",
                    dest_key="pandas_output",
                ),
            ]
        )
        qp.add_link("response_synthesis_prompt", "llm2")

        query_str = st.text_input("**Enter your question about the dataset**")

        if st.button("Submit Query"):
            # Search for relevant rows in Qdrant
            search_results = search_qdrant(query_str)

            # Show the top search results
            st.write("**Search Results from Qdrant:**")
            for result in search_results:
                st.write(result.payload["row_data"])

            # Run the query pipeline (if needed for additional processing)
            response = qp.run(query_str=query_str)
            st.write("**Response from LLM:**")
            st.write(response)


if __name__ == "__main__":
    main()
