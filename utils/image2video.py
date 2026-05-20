import os
import cv2

#图像转视频工具
class Image2Video(object):
    def __init__(self,img_path):
        self.img_path = img_path
        self.save_path = './video'
    def image2video(self):

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        fps = 20
        frames = sorted(os.listdir(self.img_path))

        img = cv2.imread(os.path.join(self.img_path,frames[0]))

        img_size = (img.shape[1], img.shape[0])

        seq_name = self.img_path.split('/')[-1]

        video_path = os.path.join(self.save_path,seq_name+'.avi')
        fourcc = cv2.VideoWriter.fourcc('M','J','P','G')

        video_writer = cv2.VideoWriter(video_path, fourcc, fps, img_size)

        for frame in frames:
            f_path = os.path.join(self.img_path,frame)
            image = cv2.imread(f_path)
            video_writer.write(image)

        video_writer.release()
