

import numpy as np
import pandas as pd
import os
import re # regex

from typing import List
from bs4 import BeautifulSoup # HTMl parser
from sec_edgar_downloader import Downloader

from lxml import html
from edgar import Edgar, TXTML, Company

class FinancialReportParser(object):
    
    def parse_10K_txt_file(self, txt_file_path: str, should_parse_item_7_only: bool = True) -> List[str]:
        """
        Parses a raw 10-K .txt file.
        
        Parameters
        ----------
        txt_file_path : str
            The path to the .txt file. 
            For example => 'data\\sec_filings\\AAPL\\10-K\\0000320193-17-000070.txt'
        
        Returns
        ----------
        A parsed List of string tokens from the txt file that is passed in.
        
        """
        raw_10K_text = None
        with open(txt_file_path) as f:
            raw_10K_text = BeautifulSoup(f.read(),"html.parser")
        
        # All of the meaningful text is embedded in the `<font>` HTML tags
        html_font_tags = raw_10K_text.find_all('font')
        
        # Check if we want to only parse the text that is in Item 7 (MD&A)
        if should_parse_item_7_only:
            return self.parse_item_7_only(html_font_tags)
       
        # Else, parse the full 10K
        return self.parse_full_10K(html_font_tags)
    
        
    
    def parse_item_7_only(self, html_font_tags: List[str]) -> List[str]:
        """
        Parses raw HTML tags from `html_font_tags` and 
        returns only the Item 7 (MD&A) content.
        
        We accomplish this by parsing all of the text that is 
        inbetween the words "Item 7." and "Item 8."
        
        """
        start_text_tag = "Item 7."
        end_text_tag = "Item 8."
        should_append_text = False

        parsed_item_7_list = list()
        for tag in html_font_tags:
            # Extract the text content from the `<font>` HTML tag
            text = tag.get_text()
            
            # Only keep Alphanumeric characters and '.' and ','
            text = re.sub('[^A-Za-z0-9.,]+', ' ', text)
            
            # Skip empty text
            if text is None or text.isspace():
                continue
            
            if start_text_tag in text:
                should_append_text = True

            if end_text_tag in text:
                should_append_text = False

            if should_append_text:
                parsed_item_7_list.append(text)


        return parsed_item_7_list
    
        
    def parse_full_10K(self, html_font_tags: List[str]) -> List[str]:
        """
        Parses raw HTML tags from `html_font_tags` and returns the entire 10-K content
        
        """
        parsed_full_10K_list = list()
        for tag in html_font_tags:
            # Extract the text content from the `<font>` HTML tag
            text = tag.get_text()
            
            # Only keep Alphanumeric characters and '.' and ','
            text = re.sub('[^A-Za-z0-9.,]+', ' ', text)
            
            # Skip empty text
            if text is None or text.isspace():
                continue
                
            parsed_full_10K_list.append(text)
        return parsed_full_10K_list

class FinancialReportParserUsingEdgar(FinancialReportParser):

    def parse_10K_txt_file(self, txt_file_path: str, should_parse_item_7_only: bool = True) -> str:
        """
        Parses a raw 10-K .txt file.

        Parameters
        ----------
        txt_file_path : str
            The path to the .txt file.
            For example => 'data\\sec_filings\\AAPL\\10-K\\0000320193-17-000070.txt'

        Returns
        ----------
        A full text containing the text required.

        """
        raw_10K_html_text = None
        with open(txt_file_path) as f:
            raw_10K_html_text = f.read()

        # parse the full 10K
        raw_10K_text = self.parse_full_10K(html.fromstring(raw_10K_html_text))

        # Check if we want to only parse the text that is in Item 7 (MD&A)
        if should_parse_item_7_only:
            return self.parse_item_7_only(raw_10K_text)
        else:
            return raw_10K_text


    def parse_item_7_only(self, raw_10K_text: str) -> str:
        """
        Parses raw text and
        returns only the Item 7 (MD&A) content.

        We accomplish this by using a Regex ato extract the text between "Item 7." and "Item 8."

        """
        max_size = 0
        raw_7K_text = ''

        pattern = '[\\n\\r\\s]+(Item[\\n\\r\\s]?7\\..+?)[\\n\\r\\s]+Item[\\n\\r\\s]?8\\.'
        for x in re.finditer(pattern, raw_10K_text, flags=re.S | re.IGNORECASE):
            if len(x.group(1)) > max_size:
                raw_7K_text = x.group(1)
                max_size = len(x.group(1))

        if not raw_7K_text:
            raise Exception(f'7K section is empty!')

        return raw_7K_text

    def parse_full_10K(self, raw_10K_html_text: str) -> str:
        """
        Parses raw html text and returns the entire 10-K content

        """
        parsed_full_10K = TXTML.parse_full_10K(raw_10K_html_text)

        return parsed_full_10K

    
def main():
    print("Hello World!")
    
    # Get the path to the 10-K (in this case, Apple's 2017 20K)
    aapl_2017_10k_path = os.path.join('data', 'sec_edgar_filings', 'AAPL', '10-K', '0000320193-17-000070.txt')    
    parser = FinancialReportParser()
    tokens = parser.parse_10K_txt_file(aapl_2017_10k_path)
    
    # Print out first 10 tokens
    print(tokens[:10])        

    
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    