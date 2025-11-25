from langchain_community.tools import DuckDuckGoSearchRun

web_search_tool = DuckDuckGoSearchRun()


web_search_tool.description = (
    "Use this tool to search the public web for up-to-date information, "
    "facts, and news about any topic."
)

web_search_tools = [web_search_tool]
