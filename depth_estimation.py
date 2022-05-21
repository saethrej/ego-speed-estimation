import torch
import torchvision
import cv2
import urllib.request
import time

import matplotlib.pyplot as plt

input_video = "/home/jenseirik/OneDrive/ETH/master/3d-vision/project/data/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-30--13-44-30/6/video_comcro.mp4"

# read input video into a torch tensor

video_frames = torchvision.io.read_video(
            input_video, 
            start_pts=4, 
            end_pts=4 + 60/25 + 1, 
            pts_unit="sec"
        )[0][:60].float()
print(video_frames.shape)

# load the depth estimation model from github and its transforms
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

image_path = "/home/jenseirik/OneDrive/ETH/master/3d-vision/cropped_preview.png"
img1 = cv2.imread(image_path)
print(img1.shape)
images = video_frames.numpy().astype(int)


# load a frame into a c
start_transform = time.process_time()
input_batch = torch.stack(
    [transform(images[i]).squeeze(0) for i in range(images.shape[0])],
    dim = 0
)
transform_time = time.process_time() - start_transform
print("processing took {} s".format(transform_time))
print(input_batch.shape)


with torch.no_grad():
    start_pred = time.process_time()
    prediction = midas(input_batch)
    pred_time = time.process_time() - start_pred
    
    start_inter = time.process_time()
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(0),
        size=images.shape[1:3],
        mode="bicubic",
        align_corners=False
    ).squeeze()
    inter_time = time.process_time() - start_inter
    

output = prediction.numpy()
print("output shape = ", output.shape)
print("Prediction Time: {} s".format(pred_time))
print("Interpolation Time: {} s".format(inter_time))

'''
for frame in range(60):
    plt.imshow(output[frame])
    plt.show()
    time.sleep(0.5)
    plt.close()
'''