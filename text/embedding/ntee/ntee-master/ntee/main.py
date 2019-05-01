from ntee.model_reader import ModelReader
model = ModelReader('ntee_300_sentence.joblib')
x = model.get_word_vector(u'Apple')
print (x)

print ("\n\n........\n\n")

y = model.get_entity_vector(u'apple')
print (y)

print ("\n\n.%%%%%%%%%%%%%%%%%%%%%%%%\n\n")


print model.get_text_vector(u'How are you?')

print "\n\n.%##########################\n\n"

import joblib
model_obj = joblib.load('ntee_300_sentence.joblib')
print model_obj.values()


