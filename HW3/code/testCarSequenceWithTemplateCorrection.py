import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade as LK 
import os
# write your script here, we recommend the above libraries for making your animation


#######################################
# Please run testCarSequence.py first #
#######################################


# Initializing ...
video = np.load("../data/carseq.npy")
rect0 = np.array([59, 116, 145, 151])
num_frames = video.shape[2]
rects = np.load("carseqrects.npy")
rects_correction = np.zeros((num_frames, 4))
rects_correction[0,:] = rect0
report_list = [1, 100, 200, 300, 400]
epsilon = 3.5
frame0 = video[:,:,0]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.axis('off')

frames = []
for i in range(num_frames):

	# Current frame
	img_0 = video[:,:,i]
	rect = rects[i,:]
	rect_t = rects_correction[i,:]
	f = ax1.imshow(video[:,:,i], cmap='gray')
	a1 = ax1.add_patch(patches.Rectangle((rect[0], rect[3]), rect[2]-rect[0], rect[1]-rect[3],linewidth=2,edgecolor='r',facecolor='none'))
	t1 = plt.text(rect[2]-25, rect[1]-2,"Q1.3",color='r')
	a2 = ax1.add_patch(patches.Rectangle((rect_t[0], rect_t[3]), rect_t[2]-rect_t[0], rect_t[1]-rect_t[3],linewidth=2,edgecolor='b',facecolor='none'))
	t2 = plt.text(rect_t[0], rect_t[1]-2,"Q1.4",color='b')
	frames.append([f,a1,a2,t1,t2])

	# next frame w/ LK algorithm
	if i == num_frames-1:
		continue
	else:
		img_1 = video[:,:,i+1]
		p1 = LK.LucasKanade(img_0, img_1, rect_t)
		rect_t = np.array([rect_t[0]+p1[0], rect_t[1]+p1[1], rect_t[2]+p1[0], rect_t[3]+p1[1]])
		p0 = rect_t[0:2]-rect0[0:2]
		p_t = LK.LucasKanade(frame0, img_1, rect0, p0=p0)

		tmp = p_t - (rects_correction[i,0:2] - rect0[0:2])
		norm = np.linalg.norm(tmp-p1)

		if norm <= epsilon:
			p = tmp
		else: 
			p = p1	
		rect_t = np.array([rects_correction[i,0]+p[0], rects_correction[i,1]+p[1], rects_correction[i,2]+p[0], rects_correction[i,3]+p[1]])
		rects_correction[i+1,:] = rect_t


np.save("carseqrects-wcrt.npy", rects_correction)
anim = animation.ArtistAnimation(fig1, frames, interval=24, blit=True, repeat_delay=1000)
ffmpegpath = os.path.abspath("./ffmpeg/bin/ffmpeg.exe")
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpegpath
writer = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
anim.save("Q1.4.mp4", writer = writer)
plt.show()


# Generate report image
fig2, ax = plt.subplots(1, len(report_list), figsize=(60, 10))
for f in range(len(report_list)):
	ax[f].imshow(video[:,:,report_list[f]-1], cmap='gray')
	ax[f].axis('off')
	ax[f].set_title('frame {}'.format(report_list[f]),fontsize=30)
	rect = rects[report_list[f]-1]
	rect_t = rects_correction[report_list[f]-1]
	ax[f].add_patch(patches.Rectangle((rect[0], rect[3]), rect[2]-rect[0], rect[1]-rect[3],linewidth=4,edgecolor='r',facecolor='none'))
	ax[f].add_patch(patches.Rectangle((rect_t[0], rect_t[3]), rect_t[2]-rect_t[0], rect_t[1]-rect_t[3],linewidth=4,edgecolor='b',facecolor='none'))
plt.tight_layout()
fig2.savefig("Q1_4.png",dpi=400)


