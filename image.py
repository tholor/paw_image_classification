from os import listdir, rename, mkdir, remove
from os.path import isfile, join, isdir, exists
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
import math
import re
from PIL import Image

#######################################
######### Plotting ##############
#######################################
def plots_files(dir, files, cols = 4, titles=None):
    plots(ims = [image.load_img(dir+f) for f in files], cols = cols, titles=titles)
    
def plots(ims, figsize=(12,6), cols = 4, interp=False, titles=None):  
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    rows = int(max(1,math.ceil(len(ims)/cols)))
    f = plt.figure(figsize=(3*cols,3*rows))
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
    
#######################################
######### Preprocessing ###############
#######################################
#transform transparency of an image to RGB (remove alpha channel)
def convertTransparentToRGB(path):
    png = Image.open(path)
    png.load() # required for png.split()
    background = Image.new("RGB", png.size, (255, 255, 255))
    background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
    encoding = re.search(r'\.[a-zA-Z]*$', str(path))
    if not encoding:
        encoding = ".jpg" 
        remove(path)
        path = path+encoding
    background.save(path, quality=100)
    return(path)
    
def convertGrayscaleToRGB(path):
    img = Image.open(path)
    rgbimg = Image.new("RGBA", img.size)
    rgbimg.paste(img)
    remove(path)
    encoding = re.search(r'\.[a-zA-Z]*$', str(path))
    if encoding:
        path = re.sub(r'\.[a-zA-Z]*$', '.jpg',path)
    else:
        path = path+'.jpg'
    rgbimg.save(path)
    return(path)
    
#make sure that all files are proper images
def cleanImages(maindir, corrupt_images):
    folders =[o for o in listdir(maindir) if isdir(join(maindir,o))]
    n_invalid = 0
    n_samples = dict()
    for subdir in folders:
        print("clean %s" % (str(subdir)))
        basedir = maindir+subdir+'/'
        files = [f for f in listdir(basedir) if isfile(join(basedir, f))]
        n_valid = 0
        for f in files:
            img_path = basedir+str(f)
            try:
                #imread(img_path)
                cur_img = image.load_img(img_path, target_size=(299, 299))
                cur_img = image.img_to_array(cur_img)
                #transform transparency
                if(cur_img.shape[2] == 4):
                    img_path = convertTransparentToRGB(img_path)
                    cur_img = image.load_img(img_path, target_size=(299, 299))
                    cur_img = image.img_to_array(cur_img)
                #transform grayscale
                elif(cur_img.shape[2] == 1):
                    img_path = convertGrayscaleToRGB(img_path)
                    cur_img = image.load_img(img_path, target_size=(299, 299))
                    cur_img = image.img_to_array(cur_img)
                cur_img = np.expand_dims(cur_img, axis=0)
                for bad_img in corrupt_images:    
                    bad_img = image.load_img(bad_img, target_size=(299, 299))
                    bad_img = image.img_to_array(bad_img)
                    bad_img = np.expand_dims(bad_img, axis=0)
                    if np.array_equal(bad_img, cur_img):
                        print("equals corrupt image: "+str(img_path))
                        remove(img_path)
                        n_invalid += 1 
                        n_valid -= 1
                        break
                        n_valid += 1
            except:
                print("not readable: "+str(img_path))
                remove(img_path)
                n_invalid += 1
                continue
        n_samples[subdir] = n_valid
    print("removed: "+str(n_invalid))
    return(n_samples)

#move some images from training to validation directory
def createValidationSet(maindir, validation_ratio):
    print("Create validation set...")
    traindir = maindir+"train/"
    valdir = maindir+"validation/"
    folders = [o for o in listdir(traindir) if isdir(join(traindir,o))]
    for subdir in folders:
        n_samples = len([1 for f in listdir(traindir+subdir) if isfile(join(traindir+subdir, f))])
        n_validation = int(validation_ratio*n_samples)
        print(str(subdir)+": take "+str(n_validation))
        train_path = traindir+subdir   
        #create same folder for validation
        val_path = valdir+subdir
        if exists(val_path):
            #move all back to train folder
            for f in listdir(val_path):
                from_file = val_path+"/"+f
                to_file = train_path+"/"+f
                rename(from_file, to_file)
        else:
             mkdir(val_path)
        #select random files and move them
        for i in range(n_validation):
            files = [f for f in listdir(train_path) if isfile(join(train_path, f))]
            rand = np.random.randint(0,len(files))
            from_file = train_path+"/"+str(files[rand])
            to_file = val_path+"/"+str(files[rand])
            rename(from_file, to_file)
            
def removeValidationSet(maindir):
    print("Remove old validation set...")
    traindir = maindir+"train/"
    valdir = maindir+"validation/"
    folders = [o for o in listdir(valdir) if isdir(join(valdir,o))]
    for subdir in folders:
        train_path = traindir+subdir  
        val_path = valdir+subdir
        files = [f for f in listdir(val_path) if isfile(join(val_path, f))]
        for f in files:
            from_file = val_path+"/"+str(f)
            to_file = train_path+"/"+str(f)
            rename(from_file, to_file)
    return True


