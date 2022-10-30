# from datasets import load_dataset
#
# dataset = load_dataset('multilabel_bench', 'eurlex-l1', split='train')

# import json
#
# import tqdm
#
# level_1_concepts = []
# level_2_concepts = []
#
# with open('multilabel_bench/eurlex.jsonl', 'w') as out_file:
#     for subset in ['train', 'dev', 'test']:
#         with open(f'/Users/rwg642/Downloads/multi_eurlex/{subset}.jsonl') as file_1:
#             for line in tqdm.tqdm(file_1.readlines()):
#                 data = json.loads(line)
#                 data['text'] = data['text']['en']
#                 data['data_type'] = subset
#                 out_file.write(json.dumps(data) + '\n')
#                 level_1_concepts.extend(data['eurovoc_concepts']['level_1'])
#                 level_2_concepts.extend(data['eurovoc_concepts']['level_2'])
import json
#
# print()
import re

import requests
import tqdm

from data.multilabel_bench.multilabel_bench import UKLEX_CONCEPTS

concepts_dict = {}
# URL = f"https://meshb.nlm.nih.gov/treeView"
# page = requests.get(URL)
#soup = BeautifulSoup(page, "html.parser")
# list = soup.find(id="ResultsContainer")

# list = soup.find_all("ul", class_="treeItem")
# for list_element in list:
# elements = soup.find_all("span")
# for element in elements:
#     concept = element.text
#     concept_id = re.search(r'[A-Z][0-9]{1,2}', concept)
#     if concept_id:
#         concept_id = concept_id.group(0)
#         if re.match(r'[A-Z][0-9]', concept):
#             concept_id = concept[:-1] + '0' + concept[-1]
#         if concept_id in MESH_CONCEPTS['level_2']:
#             concepts_dict[concept_id] = concept.split('[')[0]
#     # element_id, element_desc = element.text.split(' ', maxsplit=1)
#     # concepts_dict[element_id] = element_desc
#
# for concept_id in MESH_CONCEPTS['level_2']:
#     try:
#         x = concepts_dict[concept_id]
#         continue
#     except:
#         print(concept_id)
#
# print()

concepts = {}
for level in UKLEX_CONCEPTS:
    concepts[level] = {}
    for concept in UKLEX_CONCEPTS[level]:
        concepts[level][concept] = (concept.title(), concept.lower())

print()

