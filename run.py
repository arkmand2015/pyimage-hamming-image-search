# import the necessary packages
from imutils import paths
#import matplotlib.pyplot as plt
import argparse
import pickle
import vptree
import time
import cv2
import os
import numpy as np
import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image,caption=title)

def dhash(image,color=0,hashSize=8):
    # convert the image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:,:,color]
    print(cv2.img_hash.colorMomentHash(image))
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(gray, (hashSize + 1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def convert_hash(h):
    # convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
    return int(np.array(h, dtype="float64"))

def hamming(a, b):
    # compute and return the Hamming distance between the integers
    return bin(int(a) ^ int(b)).count("1")

#Fetchs file from bucket

def read_file(file_path,bucket_name='atx_banana_republic'):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path)
    return content


st.title('Demo pyimagesearch')
args = {
    "tree": "vptree.pickle",
    "hashes": "hashes.pickle",
    "query": "queries/SweaterQ5.jpg",
    "distance": 100
}


st.text("[INFO] loading VP-Tree and hashes...")
tree = pickle.loads(open(args["tree"], "rb").read())
hashes = pickle.loads(open(args["hashes"], "rb").read())

image = st.file_uploader('Sube una imagen',accept_multiple_files=False)
if image is not None:
    bytes_data = image.getvalue()
    image = cv2.imdecode(np.asarray(bytearray(bytes_data), dtype=np.uint8), cv2.IMREAD_COLOR)
    b,g,r = cv2.split(image)
    image2 = cv2.merge([r,g,b])
    st.image(image2)
 
    # load the input query image

    # compute the hash for the query image, then convert it
    queryHash = dhash(image)
    queryHash = convert_hash(queryHash)

    print("[INFO] performing search...")
    start = time.time()
    results = tree.get_all_in_range(queryHash, args["distance"])
    results = sorted(results)
    end = time.time()
    st.subheader('Resultados de busqueda')
    st.text("[INFO] search took {} seconds".format(end - start))

    # loop over the results
    for (d, h) in results:
        # grab all image paths in our dataset with the same hash
        resultPaths = hashes.get(h, [])
        st.text("[INFO] {} total image(s) with d: {}, h: {}".format(
            len(resultPaths), d, h))

        # loop over the result paths
        for resultPath in resultPaths[:20]:
            # load the result image and display it to our screen
            st.image(f'https://storage.cloud.google.com/atx_banana_republic/{resultPath}',caption='Result', width=300)
            #result = cv2.imread(resultPath)
            