#!/usr/bin/env python3

import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

class ModelData(object):
    MODEL_PATH = "/home/nvidia/drone_sim/src/drone_ai/scripts/helpers/openpose/models/pose_iter_440000.caffemodel"
    DEPLOY_PATH = "/home/nvidia/drone_sim/src/drone_ai/scripts/helpers/openpose/models/pose_deploy_linevec.prototxt"
    INPUT_SHAPE = (1, 3, 1, 1)
    OUTPUT_NAME = "net_output"
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

# The Caffe path is used for Caffe2 models.
def build_engine_caffe(model_file, deploy_file):
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        # Workspace size is the maximum amount of memory available to the builder while building an engine.
        # It should generally be set as high as possible.
        builder.max_workspace_size = common.GiB(1)
        # Load the Caffe model and parse it in order to populate the TensorRT network.
        # This function returns an object that we can query to find tensors by name.
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        # For Caffe, we need to manually mark the output of the network.
        # Since we know the name of the output tensor, we can find it in model_tensors.
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        return builder.build_cuda_engine(network)

def main():
    # Set the data path to the directory that contains the trained models and test images for inference.
    caffe_model_file = ModelData.MODEL_PATH
    caffe_deploy_file = ModelData.DEPLOY_PATH

    # Build a TensorRT engine.
    engine = build_engine_caffe(caffe_model_file, caffe_deploy_file)
    with open("/home/nvidia/vgg19.engine", "wb") as f:
              f.write(engine.serialize())
   

       

if __name__ == '__main__':
    main()
