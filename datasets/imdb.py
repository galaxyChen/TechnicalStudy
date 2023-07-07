# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""IMDB movie reviews dataset."""

import datasets
from datasets.tasks import TextClassification


_DESCRIPTION = """\
Large Movie Review Dataset.
This is a dataset for binary sentiment classification containing substantially \
more data than previous benchmark datasets. We provide a set of 25,000 highly \
polar movie reviews for training, and 25,000 for testing. There is additional \
unlabeled data for use as well.\
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

_DOWNLOAD_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


class IMDBReviewsConfig(datasets.BuilderConfig):
    """BuilderConfig for IMDBReviews."""

    def __init__(self, **kwargs):
        """BuilderConfig for IMDBReviews.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(IMDBReviewsConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)

# _info方法定义数据集结构，_split_generators方法指定不同数据划分要加载的数据，_generate_examples方法实现具体的读取数据代码
class Imdb(datasets.GeneratorBasedBuilder):
    """IMDB movie reviews dataset."""

    BUILDER_CONFIGS = [
        IMDBReviewsConfig(
            name="plain_text",
            description="Plain text",
        )
    ]

    def _info(self):
        """
            info方法，定义数据集的信息，这里要对数据的字段进行定义
            
        :return:
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"text": datasets.Value("string"), "label": datasets.features.ClassLabel(names=["neg", "pos"])}
            ),
            supervised_keys=None,
            homepage="http://ai.stanford.edu/~amaas/data/sentiment/",
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):

        """
            定义了数据集的划分（split）
            这里包括训练集（datasets.Split.TRAIN）、测试集（datasets.Split.TEST）和无监督数据集（"unsupervised"）
            对于每个划分，使用 datasets.SplitGenerator 创建一个划分生成器，并指定相应的参数，如文件迭代器（files）、划分名称（split）和是否有标签（labeled）
            涉及两个参数：name和gen_kwargs
            name: 指定数据集的划分
            gen_kwargs: 指定要读取的文件的路径，与_generate_examples的入参数一致
        """

        archive = dl_manager.download(_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "test"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split("unsupervised"),
                gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "train", "labeled": False},
            ),
        ]

    def _generate_examples(self, files, split, labeled=True):
        """Generate aclImdb examples.
            生成具体的样本，使用yield
        """
        # For labeled examples, extract the label from the path.
        if labeled:
            label_mapping = {"pos": 1, "neg": 0}
            for path, f in files:
                if path.startswith(f"aclImdb/{split}"):
                    label = label_mapping.get(path.split("/")[2])
                    if label is not None:
                        yield path, {"text": f.read().decode("utf-8"), "label": label}
        else:
            for path, f in files:
                if path.startswith(f"aclImdb/{split}"):
                    if path.split("/")[2] == "unsup":
                        yield path, {"text": f.read().decode("utf-8"), "label": -1}