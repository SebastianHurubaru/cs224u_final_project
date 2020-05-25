"""
Financial statement data pipeline

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import tensorflow as tf
import tensorflow_text as text

import os
import json
from sec_edgar_downloader import Downloader

import tensorflow_datasets.public_api as tfds

from parsers import FinancialReportParser


class FinancialStatementDatasetBuilder(tfds.core.GeneratorBasedBuilder):

    def __init__(self, args):

        self.args = args

        self.VERSION = tfds.core.Version(self.args.dataset_version)
        self.MANUAL_DOWNLOAD_INSTRUCTIONS = "Dataset already downloaded manually"

        super(tfds.core.GeneratorBasedBuilder, self).__init__()

        self.dl = Downloader(self.args.download_path)

        self.parser = FinancialReportParser()

        self.tokenizer = text.WhitespaceTokenizer()

    def _info(self):

        return tfds.core.DatasetInfo(
            builder=self,

            description=("Financial statements data."),

            features=tfds.features.FeaturesDict({
                "documents": tfds.features.Tensor(
                    dtype=tf.string, shape=(self.args.number_of_periods,)
                ),
                "label": tfds.features.ClassLabel(num_classes=2),
            }),

            supervised_keys=("documents", "label"),

            homepage="https://xxx",

            citation=r"""@article{my-awesome-dataset-2020,
                                  author = {Hurubaru, Sebastian},"}""",
        )

    def _split_generators(self, dl_manager):

        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "input_dir": os.path.join(self.args.input_dir, 'train')
                },
            ),

            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "input_dir": os.path.join(self.args.input_dir, 'dev')
                },
            ),

            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "input_dir": os.path.join(self.args.input_dir, 'test')
                },
            )
        ]

    def _generate_examples(self, input_dir):

        # Get the content of the dataset file
        dataset = tf.data.experimental.make_csv_dataset(
            os.path.join(input_dir, self.args.company_files),
            batch_size=1,
            column_defaults=[tf.string, tf.string, tf.string, tf.int32],
            label_name='label',
            na_value="?",
            num_epochs=1,
            ignore_errors=True
        )

        for company_info, label in dataset:

            ciks = company_info['cik'].numpy()[0].decode('utf-8').split(';')
            ciks.sort(reverse=True, key=lambda cik: int(cik))
            end_date = company_info['end_date'].numpy()[0].decode('utf-8')

            try:

                documents = []

                # For multiple CIKs take in the descending order the last args.number_of_periods 10-K reports
                for cik in ciks:

                    cik_folder = os.path.join(
                        os.path.expanduser(self.args.download_path),
                        'sec_edgar_filings',
                        cik.strip().lstrip("0"),
                        '10-K')

                    # Download if and only if the directories do not exist
                    if (os.path.exists(cik_folder) is False):
                        self.dl.get("10-K",
                                    cik,
                                    before_date=end_date,
                                    num_filings_to_download=self.args.number_of_periods)

                    for r, d, f in os.walk(cik_folder):
                        for file in f:
                            if '.txt' in file:
                                documents.append(
                                    tf.convert_to_tensor(
                                        self.parser.parse_10K_txt_file(os.path.join(r, file)),
                                        dtype=tf.string))

                if len(documents) < self.args.number_of_periods:
                    raise Exception(f'Could not retrieve {self.args.number_of_periods} 10-K records for {cik}')

                yield cik, {
                    'documents': tf.stack(documents)[:self.args.number_of_periods],
                    'label': label.numpy()[0]
                }

            except Exception as e:
                print(f'Exception occurred for cik {cik}: {e}')

    def _process_text_map_fn(self, text, label):

        processed_text, label = tf.py_function(self._process_text,
                                               inp=[text, label],
                                               Tout=(tf.int64, tf.int64))

        return processed_text, label

    def _process_text(self, text, label):

        # TODO: preprocess the text. Toy case to return just the tokens number

        print(f'text shape: {text.shape}')
        for i in range(text.shape[0]):
            print(f'10-k Section 7: {text[i]}')
        tokens_number = self.tokenizer.tokenize(text).shape[0]

        return (tokens_number, label)
