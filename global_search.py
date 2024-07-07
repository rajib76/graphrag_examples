import os
import pandas as pd
import tiktoken
import asyncio

from dotenv import load_dotenv
from rich import print
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch

load_dotenv()
# 1. Setup LLM
api_key = os.environ["GRAPHRAG_API_KEY"]


# llm_model = "gpt-3.5-turbo"
# llm = ChatOpenAI(
#     api_key=api_key,
#     model=llm_model,
#     api_type=OpenaiApiType.OpenAI,
#     max_retries=20,
# )


class GlobalGraphSearch():
    def __init__(self,
                 community_level=2,
                 community_report_table="create_final_community_reports",
                 entity_table="create_final_nodes",
                 entity_embedding_table="create_final_entities"):
        self.model = "gpt-3.5-turbo"
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

    def load_context_info(self):
        entity_df = pd.read_parquet(f"{self.input_dir}/{self.entity_table}.parquet")
        report_df = pd.read_parquet(f"{self.input_dir}/{self.community_report_table}.parquet")
        entity_embedding_df = pd.read_parquet(f"{self.input_dir}/{self.entity_embedding_table}.parquet")
        reports = read_indexer_reports(report_df, entity_df, self.community_level)
        entities = read_indexer_entities(entity_df, entity_embedding_df, self.community_level)

        return reports, entities

    async def run_the_search_engine(self, query):
        reports, entities = self.load_context_info()
        search_context = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,
            token_encoder=self.token_encoder,
        )
        search_context_params = {"use_community_summary": False, "shuffle_data": True, "include_community_rank": True,
                                 "min_community_rank": 0, "community_rank_name": "rank",
                                 "include_community_weight": True, "community_weight_name": "occurrence weight",
                                 "normalize_community_weight": True, "max_tokens": 3_000, "context_name": "Reports", }
        map_params = {"max_tokens": 1000, "temperature": 0.0, "response_format": {"type": "json_object"}, }
        reduce_params = {"max_tokens": 2000, "temperature": 0.0, }

        graph_search_engine = GlobalSearch(
            llm=self.llm,
            context_builder=search_context,
            token_encoder=self.token_encoder,
            max_data_tokens=12_000,
            map_llm_params=map_params,
            reduce_llm_params=reduce_params,
            allow_general_knowledge=False,
            json_mode=True,
            context_builder_params=search_context_params,
            concurrent_coroutines=10,
            response_type="multiple-page report",
            # Free form text e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

        result = await graph_search_engine.asearch(query)
        return result


if __name__ == "__main__":
    gs = GlobalGraphSearch()
    query = "Taj Mahal"
    result = asyncio.run(gs.run_the_search_engine(query))
    print(result)
