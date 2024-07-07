import asyncio
import os

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports, read_indexer_relationships, \
    read_indexer_text_units
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai import OpenAIEmbedding
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores import LanceDBVectorStore
from rich import print

load_dotenv()
api_key = os.environ["GRAPHRAG_API_KEY"]


class LocalGraphSearch():
    def __init__(self,
                 community_level=0,
                 community_report_table="create_final_community_reports",
                 entity_table="create_final_nodes",
                 entity_embedding_table="create_final_entities",
                 relationship_table="create_final_relationships",
                 text_units_table="create_final_text_units",
                 lancedb_uri="lancedb"):
        self.model = "gpt-3.5-turbo"
        self.embedding_model = "text-embedding-3-small"
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=self.model,
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.community_level = community_level
        self.input_dir = "output/20240706-171656/artifacts"
        self.community_report_table = community_report_table
        self.entity_table = entity_table
        self.entity_embedding_table = entity_embedding_table
        self.relationship_table = relationship_table
        self.text_units_table = text_units_table
        self.lancedb_uri = lancedb_uri
        self.text_embedder = OpenAIEmbedding(api_key=api_key,
                                             api_type=OpenaiApiType.OpenAI,
                                             model=self.embedding_model,
                                             max_retries=20)

    def load_context_info(self):
        entity_df = pd.read_parquet(f"{self.input_dir}/{self.entity_table}.parquet")
        report_df = pd.read_parquet(f"{self.input_dir}/{self.community_report_table}.parquet")
        entity_embedding_df = pd.read_parquet(f"{self.input_dir}/{self.entity_embedding_table}.parquet")
        relationship_df = pd.read_parquet(f"{self.input_dir}/{self.relationship_table}.parquet")
        text_unit_df = pd.read_parquet(f"{self.input_dir}/{self.text_units_table}.parquet")
        relationships = read_indexer_relationships(relationship_df)
        text_units = read_indexer_text_units(text_unit_df)
        reports = read_indexer_reports(report_df, entity_df, self.community_level)
        entities = read_indexer_entities(entity_df, entity_embedding_df, self.community_level)

        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=self.lancedb_uri)
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

        return reports, text_units, entities, relationships, description_embedding_store

    async def run_the_search_engine(self, query):
        reports, text_units, entities, relationships, description_embedding_store = self.load_context_info()
        search_context = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )
        search_context_params = {"text_unit_prop": 0.5, "community_prop": 0.1, "conversation_history_max_turns": 5,
                                 "conversation_history_user_turns_only": True, "top_k_mapped_entities": 10,
                                 "top_k_relationships": 10, "include_entity_rank": True,
                                 "include_relationship_weight": True, "include_community_rank": False,
                                 "return_candidate_context": False, "max_tokens": 12_000, }
        llm_params = {"max_tokens": 2_000, "temperature": 0.0, }

        graph_search_engine = LocalSearch(
            llm=self.llm,
            context_builder=search_context,
            token_encoder=self.token_encoder,
            llm_params=llm_params,
            context_builder_params=search_context_params,
            response_type="multiple paragraphs",
        )

        result = await graph_search_engine.asearch(query)
        return result


if __name__ == "__main__":
    gs = LocalGraphSearch()
    query = "Taj Mahal"
    result = asyncio.run(gs.run_the_search_engine(query))
    print(result)
