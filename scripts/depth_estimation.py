import torch
import torchvision
import cv2
import time

import matplotlib.pyplot as plt

input_video = "/cluster/project/infk/courses/252-0579-00L/group10/ego-speed-estimation/data/comma2k19/Chunk_1/b0c9d2329ad1606b|2018-07-29--12-02-42/27/video_comcro.mp4"

# read input video into a torch tensor

video_frames = torchvision.io.read_video(
            input_video, 
            start_pts=4, 
            end_pts=4 + 60/25 + 1, 
            pts_unit="sec"
        )[0][:60].float()
print(video_frames.shape)

# load the depth estimation model from github and its transforms
torch.hub.set_dir("./torch_hub")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

images = video_frames.numpy().astype(int)


# load a frame into a c
start_transform = time.perf_counter()
input_batch = torch.stack(
    [transform(images[i]).squeeze(0) for i in range(images.shape[0])],
    dim = 0
).to(device)
transform_time = time.perf_counter() - start_transform
print("processing took {} s".format(transform_time))
print(input_batch.shape)


with torch.no_grad():
    start_pred = time.perf_counter()
    prediction = midas(input_batch)
    pred_time = time.perf_counter() - start_pred
    
    start_inter = time.perf_counter()
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(0),
        size=images.shape[1:3],
        mode="bicubic",
        align_corners=False
    ).squeeze()
output = prediction.cpu().numpy()
inter_time = time.perf_counter() - start_inter

print("output shape = ", output.shape)
print("Prediction Time: {} s".format(pred_time))
print("Interpolation Time: {} s".format(inter_time))
