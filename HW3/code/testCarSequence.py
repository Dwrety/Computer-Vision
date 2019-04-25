import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade as LK 
import os
# write your script here, we recommend the above libraries for making your animation


# Initializing ...
video = np.load("../data/carseq.npy")
rect0 = np.array([59, 116, 145, 151])
num_frames = video.shape[2]
rects = np.zeros((num_frames, 4))
rects[0,:] = rect0
report_list = [1, 100, 200, 300, 400]

# plotting stuff
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.axis('off')

frames = []
for i in range(num_frames):

	# Current frame
	img_0 = video[:,:,i]
	rect = rects[i,:]
	f = ax1.imshow(video[:,:,i], cmap='gray')
	a = ax1.add_patch(patches.Rectangle((rect[0], rect[3]), rect[2]-rect[0], rect[1]-rect[3],linewidth=2,edgecolor='r',facecolor='none'))
	t = plt.text(rect[2]-25, rect[1]-2,"Q1.3",color='r')
	frames.append([f,a,t])

	# next frame w/ LK algorithm
	if i == num_frames-1:
		continue
	else:
		img_1 = video[:,:,i+1]
		p = LK.LucasKanade(img_0, img_1, rect)
		rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
		rects[i+1,:] = rect

np.save("carseqrects.npy", rects)
anim = animation.ArtistAnimation(fig1, frames, interval=24, blit=True, repeat_delay=1000)
ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
anim.save("Q1.3.mp4", writer = writer)
plt.show()


# Generate report image
fig2, ax = plt.subplots(1, len(report_list), figsize=(60, 10))
for f in range(len(report_list)):
	ax[f].imshow(video[:,:,report_list[f]-1], cmap='gray')
	ax[f].axis('off')
	ax[f].set_title('frame {}'.format(report_list[f]),fontsize=30)
	rect = rects[report_list[f]-1]
	ax[f].add_patch(patches.Rectangle((rect[0], rect[3]), rect[2]-rect[0], rect[1]-rect[3],linewidth=4,edgecolor='r',facecolor='none'))
plt.tight_layout()
fig2.savefig("Q1_3.png",dpi=400)


