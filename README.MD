# Web Scraper and Metadata Analyzer

This project provides tools for web scraping, data cleaning, and metadata analysis using Ollama with qwen2:1.5b and Crawl4AI. It enables efficient content extraction techniques.

## Features

- Web scraping with crawl4ai using multiagent agents for content extraction
- Data cleaning for extracted metadata
- Metadata analysis for SEO insights

## Tools

1. **WebScraperTool**: Extracts title, description, and keywords from web pages.
2. **DataCleanerTool**: Cleans and formats the extracted metadata.
3. **MetadataAnalyzerTool**: Analyzes the cleaned metadata to extract insights.

## Requirements

- Python 3.10+
- crawl4ai
- pydantic
- praisonai_tools

## Installation

1. Clone the repository: https://github.com/varunsaagar/crawlwithagents.git and cd crawlwithagents


2. Create and activate a virtual environment:

python -m venv venv source venv/bin/activate 

On Windows, use venv\Scripts\activate.bat

On Mac, use venv/bin/activate

On Linux, use source venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Run the main script:

python crawler.py 
 
or

The tools can be executed from the command line. Here's how to run each tool:

### Web Scraper Tool

This tool scrapes the title, description, and keywords from a specified URL.

bash python -m praisonai_tools.tools.web_scraper_tool.web_scraper_tool --url "https://example.com"


### Data Cleaner Tool

This tool cleans and formats the raw metadata extracted by the Web Scraper Tool.

bash python -m praisonai_tools.tools.data_cleaner_tool --input "path/to/raw_data.json"


### Metadata Analyzer Tool

This tool analyzes the cleaned metadata to provide insights such as title length, description length, and keyword analysis.

bash python -m praisonai_tools.tools.metadata_analyzer_tool --input "path/to/cleaned_data.json"


## Configuration

The tools are configured to work with the `crawl4ai` library and the Ollama model. You can modify the extraction strategy and other parameters in the script.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Credits
Special thanks to the following resources and individuals:
- **[Crawl4ai](https://github.com/unclecode/crawl4ai)** for the library.
- **[Agents Wraper for Crawl4ai](https://github.com/MervinPraison/PraisonAI)** for the agents wrapper.
- **[Ollama](https://github.com/ollama/ollama)** for the Ollama Framework.

## Contact

For any queries or technical support, please contact [varunsaagar](mailto:varunsaagar.s@gmail.com).




