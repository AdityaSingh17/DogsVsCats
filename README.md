# Dogs vs Cats
An extracurricular project to train an image classification deep neural network on Tensorflow and Keras to classify the given input image as a **Dog** or **Cat**.

Data set used: https://www.kaggle.com/c/dogs-vs-cats

## Website
Application Website: http://dogorcat.pythonanywhere.com/

### Usage
* Visit the website [here](http://dogorcat.pythonanywhere.com/).
* Click on `Browse` button to select an image from your local machine.
* Click on `Upload` button to upload the selected image on the server.
* Sit back and wait for the prediction.

## API
API Endpoint: http://dogorcat.pythonanywhere.com/query

### Usage
* Check file [api.py](https://github.com/AdityaSingh17/DogsVsCats/blob/master/api.py) for a working example of API.
* Specify URL on the line that states ```URL = "<URL HERE>"```
* Example:  ```URL = "https://i.redd.it/39w0xd9ersxz.jpg"```
* Run file to see the result.

### Note
This website is hosted on [pythonanywhere](https://www.pythonanywhere.com/) servers. Some of the image hosting websites are not in the **White list** of the host. 

You may see the message: `Pythonanywhere servers will not let us access that URL! :( Please try a different URL.`, while trying to use the API,
  * You can either download the image and upload them [here](http://dogorcat.pythonanywhere.com/).
  * Or you can try to provide URL of image hosted on websites which are listed [here](https://www.pythonanywhere.com/whitelist/).

