import os
import tensorflow as tf
import tf2onnx
from architecture import InceptionResNetV2

if __name__ == '__main__':
    """ weights can be downloaded from https://drive.google.com/drive/folders/1scGoVCQp-cNwKTKOUqevCP1N2LlyXU3l?usp=sharing
    Put facenet_keras_weights.h5 file in model folder
    """
    facenet_weights_path = "models/facenet_keras_weights.h5"
    onnx_model_output_path = "models/faceNet.onnx"

    if not os.path.exists(facenet_weights_path):
        raise Exception(f"Model doesn't exists in {facenet_weights_path}, download weights from \
            https://drive.google.com/drive/folders/1scGoVCQp-cNwKTKOUqevCP1N2LlyXU3l?usp=sharing")

    faceNet = InceptionResNetV2()
    faceNet.load_weights(facenet_weights_path) 

    spec = (tf.TensorSpec(faceNet.inputs[0].shape, tf.float32, name="image_input"),)
    tf2onnx.convert.from_keras(faceNet, output_path=onnx_model_output_path, input_signature=spec)