# crawl\lib\python3.10\site-packages\praisonai_tools\tools\web_scraper_tool\web_scraper_tool.py
import json
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from praisonai_tools import BaseTool

class PageMetadata(BaseModel):
    title: str = Field(..., description="Title of the page.")
    description: str = Field(..., description="Description or summary of the page.")
    keywords: list = Field(..., description="Keywords assigned to the page.")

class WebScraperTool(BaseTool):
    name: str = "WebScraperTool"
    description: str = "Extracts title, description, and keywords from the given webpage."

    def _run(self, url: str):
        crawler = WebCrawler(verbose=True)
        crawler.warmup()

        result = crawler.run(
            url=url,
            word_count_threshold=1,
            extraction_strategy=LLMExtractionStrategy(
                provider="ollama/qwen2:1.5b",
                api_token="no-token",
                schema=PageMetadata.model_json_schema(),
                extraction_type="schema",
                apply_chunking=False,
                instruction="""
                From the crawled content, extract the following details:
                1. Title of the page
                2. Description or summary of the page
                3. Keywords assigned to the page, which is a list of keywords.
                The extracted JSON format should look like this:
                {
                    "title": "Page Title",
                    "description": "Description or summary of the page.",
                    "keywords": ["keyword1", "keyword2", "keyword3"]
                }
                """
            ),
            bypass_cache=True,
        )
        
        result = result.extracted_content.encode('utf-8', errors='ignore').decode("unicode_escape")
        result_json = json.loads(result)
        return result_json[0]  # Return the first (and only) result

class DataCleanerTool(BaseTool):
    name: str = "DataCleanerTool"
    description: str = "Cleans and formats the extracted metadata."

    def _run(self, metadata: dict):
        # Implement cleaning logic here
        cleaned_metadata = {
            "title": metadata["title"].strip(),
            "description": metadata["description"].strip(),
            "keywords": [keyword.strip().lower() for keyword in metadata["keywords"]]
        }
        return cleaned_metadata

class MetadataAnalyzerTool(BaseTool):
    name: str = "MetadataAnalyzerTool"
    description: str = "Analyzes the cleaned metadata to extract insights."

    def _run(self, cleaned_metadata: dict):
        # Implement analysis logic here
        analysis = {
            "title_length": len(cleaned_metadata["title"]),
            "description_length": len(cleaned_metadata["description"]),
            "keyword_count": len(cleaned_metadata["keywords"]),
            "top_keywords": cleaned_metadata["keywords"][:5]  # Example: top 5 keywords
        }
        return analysis

if __name__ == "__main__":
    # Test the WebScraperTool
    scraper_tool = WebScraperTool()
    url = "https://ollama.com/blog/gemma2"  
    scraped_data = scraper_tool.run(url)
    print("Scraped Data:", json.dumps(scraped_data, indent=2))

    # Test the DataCleanerTool
    cleaner_tool = DataCleanerTool()
    cleaned_data = cleaner_tool.run(scraped_data)
    print("\nCleaned Data:", json.dumps(cleaned_data, indent=2))

    # Test the MetadataAnalyzerTool
    analyzer_tool = MetadataAnalyzerTool()
    analysis_result = analyzer_tool.run(cleaned_data)
    print("\nAnalysis Result:", json.dumps(analysis_result, indent=2))
