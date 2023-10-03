import spacy
import random
from spacy.training.example import Example
from spacy.training.iob_utils import offsets_to_biluo_tags

nlp = spacy.load("en_core_web_sm")

ner = nlp.get_pipe("ner")
ner.add_label("PLACE")
ner.add_label("DEVICE")
ner.add_label("NUMBER")

train_data = [
    ('Turn on the light in the kitchen.', {"entities": [(12, 17, "DEVICE"), (25, 32, "PLACE")]}),
    ('Dim the light in the living room.', {"entities": [(8, 13, "DEVICE")]}),
    ('The kitchen lights switch controls the room lighting.', {"entities": [(12, 18, "DEVICE"), (4, 11, "PLACE")]}),
    ('Light bulbs are energy-efficient.', {"entities": [(0, 5, "DEVICE")]}),
    ('Please adjust the Light level in kitchen.', {"entities": [(18, 23, "DEVICE"), (33, 40, "PLACE")]}),
    ('The light is too bright in kitchen.', {"entities": [(4, 9, "DEVICE"), (27, 34, "PLACE")]}),
    ('I prefer warm lights in the kitchen.', {"entities": [(14, 20, "DEVICE"), (28, 35, "PLACE")]}),
    ('Garage light fixture needs replacement.', {"entities": [(7, 12, "DEVICE"), (0, 6, "PLACE")]}),
    ('The light sensor is malfunctioning in garage.', {"entities": [(4, 9, "DEVICE"), (38, 44, "PLACE")]}),
    ('Garage light have to be on.', {"entities": [(7, 12, "DEVICE"), (0, 6, "PLACE")]}),
    ('Garage is closed', {"entities": [(0, 6, "PLACE")]}),
    ('Door to garage is open', {"entities": [(8, 14, "PLACE")]}),
    ('There are four lights in the garage.', {"entities": [(10, 14, "NUMBER"), (29, 35, "PLACE")]}),
    ('The kitchen has two lights.', {"entities": [(16, 19, "NUMBER"), (27, 33, "PLACE")]}),
    ('There are 4 lights in the garage.', {"entities": [(10, 11, "NUMBER"), (26, 32, "PLACE")]}),
    ('The kitchen has seven lights.', {"entities": [(16, 21, "NUMBER"), (4, 11, "PLACE")]}),
    ('Five lights in the garage.', {"entities": [(0, 4, "NUMBER"), (5, 11, "DEVICE"), (19, 25, "PLACE")]}),
    ('2 lights in the kitchen.', {"entities": [(0, 1, "NUMBER"), (2, 8, "DEVICE"), (16, 23, "PLACE")]}),
    ('Two lights in the Kitchen.', {"entities": [(0, 3, "NUMBER"), (4, 10, "DEVICE"), (18, 25, "PLACE")]}),
    ('There are ten lights in the kitchen.', {"entities": [(10, 13, "NUMBER"), (14, 20, "DEVICE"), (28, 35, "PLACE")]}),
]


cleaned_train_data = []
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    entities = annotations.get("entities", [])
    tags = offsets_to_biluo_tags(doc, entities)
    if "-" not in tags:
        cleaned_train_data.append((text, annotations))
    else:
        print(f"Skipping misaligned example: '{text}'")

# train model
for epoch in range(20):  
    random.shuffle(cleaned_train_data)
    for text, annotations in cleaned_train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5)  

nlp.to_disk("custom_ner_model")

custom_ner = spacy.load("custom_ner_model")
text = "One light in the kitchen."
doc = nlp(text)

entities = [(entity.text, entity.label_) for entity in doc.ents]
matrix = {}
for entity, label in entities:
    matrix[entity] = label
print(matrix)