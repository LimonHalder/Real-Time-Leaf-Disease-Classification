import itertools, time, os, numpy  as np,sys
from PIL import Image
import tensorflow as tf
import cv2
# for raspbery pi implementation
# from tflite_runtime.interpreter import Interpreter


def lite_model(interpreter,images):
  interpreter.allocate_tensors()
  interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
  interpreter.invoke()
  return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])


def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

class_names =  load_labels("labelmap.txt")

path_to_tflite = "model.tflite"

def detect_from_camera():
    ## for raspbery pi implementation
    #interpreter = Interpreter( path_to_tflite )
    interpreter = tf.lite.Interpreter( path_to_tflite )
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Image Shape (", width, ",", height, ")")

    cap = cv2.VideoCapture(0)  # 0はカメラのデバイス番号
    while True:
        # capture image
        ret, img_org = cap.read()
        #		cv2.imshow('image', img_org)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # prepara input image
        img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (244, 244))
        #img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])  # (1, 300, 300, 3)
        img = img.astype(np.float32)

        #img_array = tf.keras.preprocessing.image.img_to_array(img)
        probs_lite = lite_model( interpreter,np.expand_dims(img, axis=0)/255 )[0]
        print ( probs_lite )

        label_index= np.argmax(probs_lite)
        score = tf.nn.softmax(probs_lite)
        print(label_index)
        print(
          "This image most likely belongs to {} with a {:.2f} percent confidence."
          .format(class_names[label_index], 100 * np.max(score))
        )
        cv2.putText(
            img=img_org,
            text=class_names[label_index],
            org=(20, 30),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=(125, 246, 55),
            thickness=2
        )

        cv2.imshow('image', img_org)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
	detect_from_camera()