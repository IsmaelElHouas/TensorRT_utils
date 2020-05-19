import tensorrt as trt
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
img = cv2.imread("group.jpg")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

frameWidth = img.shape[1]
frameHeight = img.shape[0]

inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)

inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

with open("vgg19.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read())



context = engine.create_execution_context()

#convert input data to Float32
img = img.astype(np.float32)
#create output array to receive data
output = np.empty(((1, 57, 38, 50)), dtype = np.float32)

d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
#transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
#execute model
context.execute_async(bindings=bindings, stream_handle=stream.handle)

#transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
#syncronize threads
stream.synchronize()

print(output)
