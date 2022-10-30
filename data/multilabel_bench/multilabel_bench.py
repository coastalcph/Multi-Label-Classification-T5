# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MultiLabelBench: A Benchmark Dataset for Multi-Label Text Classification. Version 1.0"""

import json
import os

import datasets

MAIN_CITATION = "TBA"

_DESCRIPTION = """MultiLabelBench: A Benchmark Dataset for Multi-Label Text Classification. Version 1.0"""


MESH_CONCEPTS = {
    "level_1": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Z'],
    "level_2": ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
                'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'B01', 'B02', 'B03', 'B04', 'B05', 'C01', 'C04', 'C05', 'C06',
                'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                'C22', 'C23', 'C24', 'C25', 'C26', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D08', 'D09', 'D10', 'D12',
                'D13', 'D20', 'D23', 'D25', 'D26', 'D27', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'F01', 'F02',
                'F03', 'F04', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13',
                'G14', 'G15', 'G16', 'G17', 'H01', 'H02', 'I01', 'I02', 'I03', 'J01', 'J02', 'J03', 'K01', 'L01', 'M01',
                'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'Z01']
}

ICD9_CONCEPTS = {
    "level_1": ['001-139', '140-239', '240-279', '280-289', '290-319', '320-389', '390-459', '460-519', '520-579',
                '580-629', '630-679', '680-709', '710-739', '740-759', '760-779', '780-799', '800-999', 'E000-E999',
                'V01-V91'],
    "level_2": ['001-009', '010-018', '020-027', '030-041', '042-042', '045-049', '050-059', '060-066', '070-079',
                '080-088', '090-099', '110-118', '120-129', '130-136', '137-139', '140-149', '150-159', '160-165',
                '170-176', '179-189', '190-199', '200-209', '210-229', '230-234', '235-238', '239-239', '240-246',
                '249-259', '260-269', '270-279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289',
                '290-294', '295-299', '300-316', '317-319', '320-327', '330-337', '338-338', '339-339', '340-349',
                '350-359', '360-379', '380-389', '390-392', '393-398', '401-405', '410-414', '415-417', '420-429',
                '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508', '510-519',
                '520-529', '530-539', '540-543', '550-553', '555-558', '560-569', '570-579', '580-589', '590-599',
                '600-608', '610-612', '614-616', '617-629', '630-639', '640-649', '650-659', '660-669', '670-677',
                '680-686', '690-698', '700-709', '710-719', '720-724', '725-729', '730-739', '740', '741', '742', '743',
                '744', '745', '746', '747', '748', '749', '750', '751', '752', '753', '754', '755', '756', '757', '758',
                '759', '760-763', '764-779', '780-789', '790-796', '797-799', '800-804', '805-809', '810-819',
                '820-829', '830-839', '840-848', '850-854', '860-869', '870-879', '880-887', '890-897', '900-904',
                '905-909', '910-919', '920-924', '925-929', '930-939', '940-949', '950-957', '958-959', '960-979',
                '980-989', '990-995', '996-999', 'E000-E000', 'E001-E030', 'E800-E807', 'E810-E819', 'E820-E825',
                'E826-E829', 'E830-E838', 'E840-E845', 'E846-E849', 'E850-E858', 'E860-E869', 'E870-E876', 'E878-E879',
                'E880-E888', 'E890-E899', 'E900-E909', 'E910-E915', 'E916-E928', 'E929-E929', 'E930-E949', 'E950-E959',
                'E960-E969', 'E970-E979', 'E980-E989', 'E990-E999', 'V01-V09', 'V10-V19', 'V20-V29', 'V30-V39',
                'V40-V49', 'V50-V59', 'V60-V69', 'V70-V82', 'V83-V84', 'V85-V85', 'V86-V86', 'V87-V87', 'V88-V88',
                'V90-V90', 'V91-V91']

}

EUROVOC_CONCEPTS = {
    "level_1": ['100142', '100143', '100144', '100145', '100146', '100147', '100148', '100149', '100150', '100151',
                '100152', '100153', '100154', '100155', '100156', '100157', '100158', '100159', '100160', '100161',
                '100162'],
    "level_2": ['100163', '100164', '100165', '100166', '100167', '100168', '100169', '100170', '100171', '100172',
                '100173', '100174', '100175', '100176', '100177', '100178', '100179', '100180', '100181', '100182',
                '100183', '100184', '100185', '100186', '100187', '100188', '100189', '100190', '100191', '100192',
                '100193', '100194', '100195', '100196', '100197', '100198', '100199', '100200', '100201', '100202',
                '100203', '100204', '100205', '100206', '100207', '100208', '100209', '100210', '100211', '100212',
                '100213', '100214', '100215', '100216', '100217', '100218', '100219', '100220', '100221', '100222',
                '100223', '100224', '100225', '100226', '100227', '100228', '100229', '100230', '100231', '100232',
                '100233', '100234', '100235', '100236', '100237', '100238', '100239', '100240', '100241', '100242',
                '100243', '100244', '100245', '100246', '100247', '100248', '100249', '100250', '100251', '100252',
                '100253', '100254', '100255', '100256', '100257', '100258', '100259', '100260', '100261', '100262',
                '100263', '100264', '100265', '100266', '100267', '100268', '100269', '100270', '100271', '100272',
                '100273', '100274', '100275', '100276', '100277', '100278', '100279', '100280', '100281', '100282',
                '100283', '100284', '100285', '100286', '100287', '100288', '100289']
}

UKLEX_CONCEPTS = {
    "level_1": ['AGRICULTURE & FOOD', 'CHILDREN', 'CRIMINAL LAW', 'EDUCATION', 'ENVIRONMENT', 'EU', 'FINANCE',
                'HEALTH CARE', 'HOUSING', 'IMMIGRATION & CITIZENSHIP', 'LOCAL GOVERNMENT', 'PLANNING & DEVELOPMENT',
                'POLITICS', 'PUBLIC ORDER', 'SOCIAL SECURITY', 'TAXATION', 'TELECOMMUNICATIONS', 'TRANSPORTATION']
    ,
    "level_2": ['AGRICULTURE', 'AIR TRANSPORT', 'ANIMALS', 'BANKING', 'BROADCASTING', 'CHILDREN', 'CITIZENSHIP',
                'CRIMINAL LAW', 'DEFENCE', 'DISABLED PERSONS', 'EDUCATION', 'ELECTIONS', 'EMPLOYMENT', 'ENVIRONMENT',
                'EU', 'FINANCE', 'FIRE AND RESCUE SERVICES', 'FOOD', 'HEALTH CARE', 'HOUSING', 'IMMIGRATION',
                'INSURANCE', 'LAND REGISTRATION', 'LAND TRANSPORTATION', 'LOCAL GOVERNMENT', 'NHS', 'PLANNING',
                'POLICE', 'POLITICAL PARTIES', 'POLLUTION', 'PUBLIC ORDER', 'SOCIAL SECURITY', 'TAXATION',
                'TELECOMMUNICATIONS', 'TERRORISM', 'TRANSPORT AND WORKS', 'URBAN DEVELOPMENT', 'WASTE', 'WATER',
                'WATER TRANSPORT']
}

MAIN_PATH = 'data_files'


class MultiLabelBenchConfig(datasets.BuilderConfig):
    """BuilderConfig for MultiLabelBench."""

    def __init__(
            self,
            text_column,
            label_column,
            url,
            data_url,
            data_file,
            citation,
            label_level,
            label_classes=None,
            dev_column="dev",
            **kwargs,
    ):
        """BuilderConfig for LexGLUE.

        Args:
          text_column: ``string`, name of the column in the jsonl file corresponding
            to the text
          label_column: `string`, name of the column in the jsonl file corresponding
            to the label
          url: `string`, url for the original project
          data_url: `string`, url to download the zip file from
          data_file: `string`, filename for data set
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          multi_label: `boolean`, True if the task is multi-label
          dev_column: `string`, name for the development subset
          **kwargs: keyword arguments forwarded to super.
        """
        super(MultiLabelBenchConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_column = text_column
        self.label_column = label_column
        self.label_level = label_level
        self.label_classes = label_classes
        self.dev_column = dev_column
        self.url = url
        self.data_url = data_url
        self.data_file = data_file
        self.citation = citation


class MultiLabelBench(datasets.GeneratorBasedBuilder):
    """MultiLabelBench: A Benchmark Dataset for Multi-Label Text Classification. Version 1.0"""

    BUILDER_CONFIGS = [
        MultiLabelBenchConfig(
            name="eurlex-l1",
            description="EU Laws",
            text_column="text",
            label_column="eurovoc_concepts",
            label_level='level_1',
            label_classes=EUROVOC_CONCEPTS['level_1'],
            dev_column="dev",
            url='TBA',
            data_url=f"eurlex.tar.gz",
            data_file="eurlex.jsonl",
            citation='TBA',
        ),
        MultiLabelBenchConfig(
            name="eurlex-l2",
            description="EU Laws",
            text_column="text",
            label_column="eurovoc_concepts",
            label_level='level_2',
            label_classes=EUROVOC_CONCEPTS['level_2'],
            dev_column="dev",
            url='TBA',
            data_url=f"eurlex.tar.gz",
            data_file="eurlex.jsonl",
            citation='TBA',
        ),
        MultiLabelBenchConfig(
            name="uklex-l1",
            description="UK Laws",
            text_column="text",
            label_column="uklex_concepts",
            label_level='level_1',
            label_classes=UKLEX_CONCEPTS['level_1'],
            dev_column="dev",
            url='TBA',
            data_url=f"uklex.tar.gz",
            data_file="uklex.jsonl",
            citation='TBA',
        ),
        MultiLabelBenchConfig(
            name="uklex-l2",
            description="UK Laws",
            text_column="text",
            label_column="uklex_concepts",
            label_level='level_2',
            label_classes=UKLEX_CONCEPTS['level_1'],
            dev_column="dev",
            url='TBA',
            data_url=f"uklex.tar.gz",
            data_file="uklex.jsonl",
            citation='TBA',
        ),
        MultiLabelBenchConfig(
            name="mimic-l1",
            description="US Hospital Discharge Summaries",
            text_column="text",
            label_column="icd9_concepts",
            label_level='level_1',
            label_classes=ICD9_CONCEPTS['level_1'],
            dev_column="dev",
            url='TBA',
            data_url=f"mimic.tar.gz",
            data_file="mimic_v2.jsonl",
            citation='TBA',
        ),
        MultiLabelBenchConfig(
            name="mimic-l2",
            description="US Hospital Discharge Summaries",
            text_column="text",
            label_column="icd9_concepts",
            label_level='level_2',
            label_classes=ICD9_CONCEPTS['level_1'],
            dev_column="dev",
            url='TBA',
            data_url=f"mimic.tar.gz",
            data_file="mimic_v2.jsonl",
            citation='TBA',
        ),
        MultiLabelBenchConfig(
            name="bioasq-l1",
            description="Medical Scientific Articles",
            text_column="abstractText",
            label_column="mesh_concepts",
            label_level='level_1',
            label_classes=MESH_CONCEPTS['level_1'],
            dev_column="dev",
            url='TBA',
            data_url=f"bioasq.tar.gz",
            data_file="bioasq.jsonl",
            citation='TBA',
        ),
        MultiLabelBenchConfig(
            name="bioasq-l2",
            description="Medical Scientific Articles",
            text_column="abstractText",
            label_column="mesh_concepts",
            label_level='level_2',
            label_classes=MESH_CONCEPTS['level_1'],
            dev_column="dev",
            url='TBA',
            data_url=f"bioasq.tar.gz",
            data_file="bioasq.jsonl",
            citation='TBA',
        ),

    ]

    def _info(self):
        features = {"text": datasets.Value("string"),
                    "concepts": datasets.features.Sequence(datasets.ClassLabel(names=self.config.label_classes))}

        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + MAIN_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(os.path.join(MAIN_PATH, self.config.data_url))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.data_file), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.data_file), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, self.config.data_file),
                    "split": self.config.dev_column,
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if data["data_type"] == split:
                    yield id_, {
                        "text": data[self.config.text_column],
                        "concepts": sorted(data[self.config.label_column][self.config.label_level]),
                    }
