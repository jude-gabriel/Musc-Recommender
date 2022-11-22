from music21 import converter,instrument # or import *
file = converter.parse('ACDC.Highway_to_Hell_K.mid')
components = []
for element in file.recurse():
    components.append(element)

print(components)