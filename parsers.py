
import os
import re  # regex

from bs4 import BeautifulSoup  # HTMl parser

from lxml import html
from edgar import Edgar, TXTML, Company


class FinancialReportParser(object):

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
        with open(txt_file_path) as f:
            raw_10K_html_text = f.read()

        try:
            # Try to parse the full HTML with the edgar library
            raw_10K_text = self.parse_full_10K_edgar(html.fromstring(raw_10K_html_text))
        except Exception as e:
            print("Could not parse 10-K using edgar due to {}.  Trying beautiful soup".format(e))

            # Edgar failed; parse the full HTML using Beautiful Soup
            raw_10K_text = self.parse_full_10K_bs(raw_10K_html_text)

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

    def parse_full_10K_edgar(self, raw_10K_html_text: str) -> str:
        """
        Parses the raw `raw_10K_html_text` using Edgar and
        returns the entire 10-K content.
        """
        parsed_full_10K = TXTML.parse_full_10K(raw_10K_html_text)

        return parsed_full_10K

    def parse_full_10K_bs(self, raw_10K_html_text: str) -> str:
        """
        Parses the raw `raw_10K_html_text` using Beautiful Soup and \
        returns the entire 10-K content.

        """
        raw_10K_text = BeautifulSoup(raw_10K_html_text, "html.parser")

        # All of the meaningful text is embedded in the `<font>` HTML tags
        html_font_tags = raw_10K_text.find_all('font')

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
        return " ".join(parsed_full_10K_list)


def get_longest_range(list_of_indexes):

    longest_diff = -1
    longest_index = 0
    for index, value in enumerate(list_of_indexes):
        start = value[0]
        end = value[1]

        diff = end - start
        if diff > longest_diff:
            longest_diff = diff
            longest_index = index

    return longest_index
