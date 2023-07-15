import cv2
import matplotlib.pyplot as plt
import numpy as np

protoPath = "C:/Users/slorate/Desktop/py/deploy.prototxt"
modelPath = "C:/Users/slorate/Desktop/py/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

def _HED(img):
    img = cv2.imread(r"C:\Users\slorate\Desktop\py\-6+5_Cut.bmp")
    plt.imshow(img)
    (H, W) = img.shape[:2]

    mean_pixel_values= np.average(img, axis = (0,1))
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             #mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             mean=(105, 117, 123),
                             swapRB= False, crop=False)

    #View image after preprocessing (blob)
    blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)
    #plt.imshow(blob_for_plot)

    # set the blob as the input to the network and perform a forward pass
    # to compute the edges
    net.setInput(blob)
    hed = net.forward()
    hed = hed[0,0,:,:]  #Drop the other axes 
    #hed = cv2.resize(hed[0, 0], (W, H))
    #hed = (255 * hed).astype("uint8")  #rescale to 0-255

    #plt.title("HED")
    #plt.imshow(hed, cmap='gray')
    #plt.show()
    return hed
