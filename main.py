from trafficSigns.data import datasets_preparing as dataset

label_names_file = "data/label_names.csv"

dataset.label_text(label_names_file)
print(dataset.load_rgb_data("data/data1.pickle"))