import pickle

txt = "hi"
pickle_out = open("resources/test.pickle", "wb")
pickle.dump(txt, pickle_out)
pickle_out.close()

pickle_in = open("resources/test.pickle", "rb")
text_in = pickle.load(pickle_in)
print(text_in)
