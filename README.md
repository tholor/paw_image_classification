# PAW 2017 Talk: "ML meets Ad Creation"
Code of my PAW 2017 conference talk. Retraining the Inception v3 Architecture for your own multilabel problem. 

# How to run it? 
1. Create the folders "data/images/train" and "data/images/validation" and place your images in subfolders named after one of the labels (e.g. "bottle")
2. Use my keras fork (or just copy the file keras/preprocessing/image.py) to allow for passing a dictionary with the "multilabels" to the keras ImageDataGenerator: https://github.com/tholor/keras/tree/multilabel-image-generator
2. Create this dictionary containing the multilabels, with key = image_path and value = numpy array of dummy encoded labels and pickle it as "data/meta/all_y_labels.p"
Example: 
```
{'vegetables/vegetables_854.jpg': array([ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
 'wallet/wallet2270_000268.jpg': array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), 
...} 
```

4. Create a dictionary for translation between object name and integer. Pickle it as "data/meta/classes_dict.p"
```
Example:
{'airplane': 35,
 'baby': 63,
 'beach': 74,
 'bed': 12,
 'bike': 62,
 'boat': 17,
 'book': 27,
 'bottle': 59,
 'brand_logo': 36,
 'business_man': 61,
 'camera': 0,
 'cat': 34,
 'chair': 4,
 'city': 8,
...}
```
5. Start the jupyter notebook, adjust the parameters batch_size, epochs and deep_epochs
6. Have fun classifying your images!
