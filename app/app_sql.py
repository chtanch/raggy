import streamlit as st
import pandas as pd
from pathlib import Path
import json
import re
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer
from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core.query_pipeline import QueryPipeline as QP, InputComponent
from llama_index.core.llms import ChatResponse
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import FnComponent


# Reuse existing functions
def sanitize_column_name(col_name):
    return re.sub(r"\W+", "_", col_name)


def create_table_from_dataframe(df, table_name, engine, metadata_obj):
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]
    table = Table(table_name, metadata_obj, *columns)
    metadata_obj.create_all(engine)
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


def parse_response_to_sql(response: ChatResponse) -> str:
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
    if response.startswith("SQLQuery:"):
        response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


def get_table_context_str(table_schema_objs, sql_database):
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(table_schema_obj.table_name)
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context
        context_strs.append(table_info)
    return "\n\n".join(context_strs)


def main():
    st.title("Query CSV using Text-To-SQL")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Data preview:**")
        st.dataframe(df.head())

        engine = create_engine("sqlite:///:memory:")
        metadata_obj = MetaData()

        table_name = "uploaded_data"
        create_table_from_dataframe(df, table_name, engine, metadata_obj)

        sql_database = SQLDatabase(engine)
        table_node_mapping = SQLTableNodeMapping(sql_database)

        table_schema_obj = SQLTableSchema(
            table_name=table_name, context_str="This table contains uploaded CSV data."
        )

        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        llm = Ollama(model="llama3")
        Settings.llm = llm

        obj_index = ObjectIndex.from_objects(
            [table_schema_obj],
            table_node_mapping,
            VectorStoreIndex,
        )

        obj_retriever = obj_index.as_retriever(similarity_top_k=1)
        sql_retriever = SQLRetriever(sql_database)

        table_parser_component = FnComponent(
            fn=lambda x: get_table_context_str(x, sql_database)
        )
        sql_parser_component = FnComponent(fn=parse_response_to_sql)

        text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
            dialect=engine.dialect.name
        )

        response_synthesis_prompt_str = (
            "Given an input question, synthesize a response from the SQL Response.\n"
            "Query: {query_str}\n"
            "SQL: {sql_query}\n"
            "SQL Response: {context_str}\n"
            "Response: "
        )

        response_synthesis_prompt = PromptTemplate(
            response_synthesis_prompt_str,
        )

        qp = QP(
            modules={
                "input": InputComponent(),
                "table_retriever": obj_retriever,
                "table_output_parser": table_parser_component,
                "text2sql_prompt": text2sql_prompt,
                "text2sql_llm": llm,
                "sql_output_parser": sql_parser_component,
                "sql_retriever": sql_retriever,
                "response_synthesis_prompt": response_synthesis_prompt,
                "response_synthesis_llm": llm,
            },
            verbose=True,
        )

        qp.add_chain(["input", "table_retriever", "table_output_parser"])
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        qp.add_chain(
            ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
        )
        qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        qp.add_link(
            "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
        )
        qp.add_link(
            "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
        )
        qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

        query_str = st.text_input("**Enter your question:**")
        if query_str:
            response = qp.run(query_str=query_str)
            st.write("**Response:**")
            st.write(response.message.content)


if __name__ == "__main__":
    main()
