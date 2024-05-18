This script is designed to build a dataset of prompts as described in the report. By adjusting the source of the data (Taiga or Twitter), users can generate prompts from Taiga's social media segment or Russian Troll Tweets. 

## Setup Instructions

Before running the `main.py` script, make sure you have completed the following steps:

1. **Data Files**:
   - Ensure that the `social.tar` archive and the `twitter` folder are in the same directory as your script. social.tar is archive with files from Taiga's [social media segment](https://tatianashavrina.github.io/taiga_site/downloads) and twitter is folder with data from [Kaggle](https://www.kaggle.com/datasets/fivethirtyeight/russian-troll-tweets)


2. **Configure API Key**:
   - You need to get an API key from Google to use Perspective API. You can find instruction [here](https://developers.perspectiveapi.com/s/docs-get-started?language=en_US)
   - Open the `process_text.py` file.
   - Find the line:
     ```python
     API_KEY = ""
     ```
     
   - Replace the empty quotes `""` with your personal API key:
     ```python
     API_KEY = "your_api_key_here"
     ```

4. **Install Dependencies**:
   - Before running the script, install all required Python packages by running the following command:
     
     ```bash
     pip install -r requirements.txt
     ```

5. **Run the Script**:
   - To process files from Taiga, use the command:
     
     ```bash
     python main.py --taiga
     ```
   - If you do not use the `--taiga` flag, the script will default to processing files from Twitter.
   
