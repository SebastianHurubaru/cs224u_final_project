"""
Financial statement data pipeline

Author:
    Sebastian Hurubaru (hurubaru@stanford.edu)
"""

import tensorflow as tf

import os
from sec_edgar_downloader import Downloader

import tensorflow_datasets.public_api as tfds

from parsers import FinancialReportParser
from text_processors import get_text_processor

#for debugging
# import pydevd

class FinancialStatementDatasetBuilder(tfds.core.GeneratorBasedBuilder):

    def __init__(self, args, log):

        self.args = args
        self.log = log

        self.VERSION = tfds.core.Version(self.args.dataset_version)
        self.MANUAL_DOWNLOAD_INSTRUCTIONS = "Dataset already downloaded manually"

        super(tfds.core.GeneratorBasedBuilder, self).__init__()

        self.dl = Downloader(self.args.download_path)

        self.parser = FinancialReportParser()
        self.text_processor = get_text_processor(args.model)(args)

    def _info(self):

        return tfds.core.DatasetInfo(
            builder=self,

            description=("Financial statements data."),

            features=tfds.features.FeaturesDict({
                "documents": tfds.features.Tensor(
                    dtype=tf.string, shape=(self.args.number_of_periods,)
                ),
                "label": tfds.features.Tensor(
                    dtype=tf.int64, shape=(2,)
                )
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
                    'label': [1, 0] if label.numpy()[0] == 0 else [0, 1]
                }

            except Exception as e:
                self.log.error(f'Exception occurred for cik {cik}: {e}')

    def _process_text_map_fn(self, text, label):
        processed_text, label = tf.py_function(self._process_text,
                                               inp=[text, label],
                                               Tout=(tf.float32, tf.int64))
        return processed_text, label

    def _process_text(self, text, label):

        # To allow debugging in the combined static eager mode
        # pydevd.settrace(suspend=True)

        # Process the text
        processed_text = self.text_processor.process_text(text)

        return (processed_text, label)
