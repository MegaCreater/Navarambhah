import sys,os,requests                                          # load sys, os and requests
from PIL import UnidentifiedImageError                          # load invalid image error
import tensorflow as tf                                         # load tensorflow

# How to run / execute -> $python {sys.argv[0]} global | local image_address

image_size:tuple=(299,299)                                                  # input image size
model_address:str='food_Xception'                                           # model address
class_labels:list=['Bread', 'Dairy product', 'Dessert', 'Egg',
                   'Fried food', 'Meat', 'Noodles-Pasta', 'Rice',
                   'Seafood', 'Soup', 'Vegetable-Fruit']                    # list of class names


# make function to read image
def readImage(image_address:str,target_size:tuple,response=None)->tuple:

    ''' function to read image '''
    
    try:                                                                    # try to read image
        image=tf.keras.utils.load_img(image_address,color_mode='rgb',target_size=target_size,interpolation='nearest')# make read image
        image_array=tf.keras.utils.img_to_array(image,dtype=None)           # pre-processed image
        if (len(image_array.shape)==3) and ((image_array.shape[-1]==3) or (image_array.shape[-1]==4)):# if image if color image of 4 or 3 channels
            if image_array.shape[-1]==4:image_array=image_array[:,:,:3]     # if image if of 4 channels (RGBA) convert to 3 channels (RGB)
            image_batch=tf.expand_dims(image_array,axis=0)                  # convert image to batch
            condition=True                                                  # make add success condition True
        else:raise ValueError(f'Given image must be a color image of 3 channels but got an image of shape - {image_array.shape}')# if image is of invalid channels          
    except FileNotFoundError:                                               # if file not found or failed to download the file
        print(f'FileNotFoundError! No such file found at address - {argvs[2]}!')# print info 
        image_batch=None                                                    # set image batch to None
        condition=True                                                      # make add success condition True
    except UnidentifiedImageError:                                          # if the image is an invalid image (Failed by Pillow to read)
        print(f'Invalid image error! Given image at address - {argvs[2]} is invalid.')# print info 
        image_batch=None                                                    # set image batch to None
        condition=True                                                      # make add success condition True
    finally:                                                                # delete the file if it is downloaded
        if str(argvs[1]).casefold()=='global'.casefold() and response.status_code==200:# check and delete 
            os.remove(image_address)                                        # if the file is downloaded -> delete the downloaded file
    return image_batch,condition                                            # return image batch and condition


if __name__=='__main__':                                                    # run under the main scope

    # main scope : How to run / execute -> $python {sys.argv[0]} global | local image_address
    
    try:model=tf.keras.models.load_model(model_address,compile=False)       # try to load the model
    except Exception as error:raise ValueError(f'Failed to load the model at "{model_address}"!!! Error - {error}')# else raise an error if failed                                                 
    if len(sys.argv)==3:                                                    # make a copy of arguments
        argvs=sys.argv.copy()                                               # make copy of arguments list  
        if not (argvs[1].casefold() in ['global'.casefold(), 'local'.casefold()]):# if argument [1] is not global or local
            raise ValueError(f'argv[1] must be global or local, but got {argvs[1]}'# print info
                             f'How to run / execute -> $python {sys.argv[0]} global | local image_address')# print help
        if (argvs[1].casefold()=='local'.casefold()) and (not(os.path.isfile(argvs[2]))):# if argument [1] is not global or local
            raise ValueError(f'No image file found locally at the given address at {argvs[2]}.'# print info
                             f'How to run / execute -> $python {sys.argv[0]} global | local image_address')# print help
        if str(argvs[1]).casefold()=='global'.casefold():                   # if the image has to be downloaded
            response=requests.get(argvs[2],params=None,stream=True)         # download the file
            image_address='image.data'                                      # set the image address
            if response.status_code==200:                                   # check if got response - OK
                with open('image.data',mode='wb') as image_file:            # save the image as a temporary file
                    image_file.write(response.raw.data)                     # read the image
                try:                                                        # try to read the image
                    image_batch,condition=readImage(image_address,image_size,response)# read the image
                    if condition:                                           # if the condition is True
                        image_batch=(image_batch-tf.reduce_min(image_batch))/(tf.reduce_min(image_batch)-tf.reduce_min(image_batch))# make preprocess image #- scale between 0 and 1
                        prediction=model.predict(image_batch,verbose=0)[0]  # make a prediction
                        class_name=class_labels[tf.argmax(prediction)]      # get the class name
                        print(f'I think it is a "{class_name}" or with a probability of {prediction[round(tf.argmax(prediction))]*100,2}%')# make print results
                    else:raise ValueError(f'Failed to load the image from the given address - {argvs[2]}')# else raise an error if failed
                except FileNotFoundError:print(f'FileNotFoundError! Failed requested file not found!!!')# except all exceptions - print error for File Not Found Error 
                except UnidentifiedImageError:print(f'Invalid image error! Given image at address - {argvs[2]} is invalid.')# if the image is an invalid image
            else:raise ValueError(f'Failed to load the image from the given address - {argvs[2]}')# else raise an error if failed
        elif str(argvs[1]).casefold()== 'local'.casefold():                 # if the image has to be read from the local directory
            image_address=argvs[2]                                          # set image address
            try:                                                            # make try to read image
              image_batch,condition=readImage(image_address,image_size)     # read image
              if condition:                                                 # if condition is true
                    image_batch=(image_batch-tf.reduce_min(image_batch))/(tf.reduce_min(image_batch)-tf.reduce_min(image_batch))# make preprocess image #- scale between 0 and 1
                    prediction=model.predict(image_batch,verbose=0)[0]      # make prediction
                    class_name=class_labels[tf.argmax(prediction)]          # get class name
                    print(f'I thinks it is a "{class_name}" or with probability of {round(prediction[tf.argmax(prediction)]*100,2)}%')# make print
              else:raise ValueError(f'Failed to load image from given address - {argvs[2]}')# else raise error if failed
            except FileNotFoundError:print(f'FileNotFoundError! Failed requested file not found!!!')# except all exception - make print info
            except UnidentifiedImageError:print(f'Invaild image error! Given image at address - {argvs[2]} is invaild.')# if image is invalid image
    else:                                                                   # else error
        raise ValueError(f'Got invalid arguments! How to run / excute -> $python {sys.argv[0]}  global | local image_address')# print help
