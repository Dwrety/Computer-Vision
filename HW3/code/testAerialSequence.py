import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from SubtractDominantMotion import SubtractDominantMotion
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
import cv2
import time 
import matplotlib
import os
# write your script here, we recommend the above libraries for making your animation

# Initializing ...
video = np.load("../data/aerialseq.npy")
num_frames = video.shape[2]
report_list = [30, 60, 90, 120]
H,W = video.shape[0:2]
function = InverseCompositionAffine

# structure = np.array([[0,0,1,0,0],
# 					  [0,1,1,1,0],
# 					  [1,1,1,1,1],
# 					  [0,1,1,1,0],
# 					  [0,0,1,0,0]])
# structure = np.array([[0,1,0],[1,1,1],[0,1,0]])

# plotting stuff
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.axis('off')
start = time.time()
frames = []
for i in range(num_frames-1):

	frame_0 = video[:,:,i]
	frame_1 = video[:,:,i+1]
	mask = SubtractDominantMotion(frame_0, frame_1, func=function)
	mask_img = np.zeros((H,W,3))
	mask_img[:,:,2] = mask.astype(float)*0.5
	frame_0 = np.stack([frame_0,frame_0,frame_0], axis=2)
	img = frame_0 + mask_img
	f2 = ax1.imshow(img)
	frames.append([f2])

print("The algorithm {} took {} seconds to run".format(function.__name__, time.time()-start))
anim = animation.ArtistAnimation(fig1, frames, interval=30, blit=True, repeat_delay=1000)
ffmpegpath = os.path.join("ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
anim.save("airair.mp4", writer = writer)
plt.show()


fig2, ax = plt.subplots(1, len(report_list), figsize=(60, 10))

for f in range(len(report_list)):
	frame = video[:,:,report_list[f]-1]
	frame_next = video[:,:,report_list[f]]
	mask = SubtractDominantMotion(frame, frame_next)
	mask_img = np.zeros((H,W,3))
	mask_img[:,:,2] = mask.astype(float)*0.5
	frame = np.stack([frame,frame,frame], axis=2)
	img = frame + mask_img
	ax[f].imshow(img)
	ax[f].axis('off')
	ax[f].set_title('frame {}'.format(report_list[f]),fontsize=30)
plt.tight_layout()
fig2.savefig("Q3_3I.png",dpi=400)