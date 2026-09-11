from src.tools.tools import web_search
from src.tools.tools import scrape_url
from src.pipelines.pipeline import run_research_pipeline
# result = web_search.invoke({
#     "query": "What is the latest news on AI?"
# })

# print(result)

# scrapped_content = scrape_url.invoke({
#     "url": "https://www.youtube.com/watch?v=_Fi4cpKCXss"
# })
# print(scrapped_content)

topic="The impact of AI on the job market in 2026"
run_research_pipeline(topic)
    