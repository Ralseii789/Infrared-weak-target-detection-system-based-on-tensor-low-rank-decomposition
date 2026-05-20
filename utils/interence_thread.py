import os
import cv2
from PIL import ImageDraw
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image
import numpy as np
import torch
from PyQt6.QtGui import QImage
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
from skimage import measure
from model.net import Net
from utils.utils import PadImg
import time
import pickle
#推理实现类

class InferenceThread(QThread):
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int)
    error_report = pyqtSignal(str)
    image_return = pyqtSignal(QImage)
    fps_count_signal = pyqtSignal(int)
    def __init__(self, paths=None, model_name='NUDT', pth_dir=None, device='cuda', result_dir='./result',run_type=0):
        super(InferenceThread, self).__init__()
        self.paths = paths
        self.model_name = model_name
        self.pth_dir = pth_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.result_dir = result_dir
        self.model = None
        self.img_norm_cfg = None
        self.run_type = run_type
        self.is_running = True
        self.video = None
        self.result_video_dir = './result_video'
        self.video_writer = None

    def safe_load_weights(self, weight_path, map_location=None):
        """
        兼容 NumPy 2.x 保存的权重文件，在 NumPy 1.x 环境中加载。
        """

        class RenameUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith('numpy._core'):
                    module = module.replace('numpy._core', 'numpy.core')
                return super().find_class(module, name)

        with open(weight_path, 'rb') as f:
            obj = RenameUnpickler(f).load()

        # 如果指定了设备，手动将张量移动到目标设备
        if map_location is not None:
            def map_tensor(obj):
                if torch.is_tensor(obj):
                    return obj.to(map_location)
                elif isinstance(obj, dict):
                    return {k: map_tensor(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [map_tensor(v) for v in obj]
                else:
                    return obj

            obj = map_tensor(obj)
        return obj

        with open(weight_path, 'rb') as f:
            return RenameUnpickler(f).load()
    #加载模型
    def load_model(self):
        try:
            from utils.utils import get_img_norm_cfg
            self.img_norm_cfg = get_img_norm_cfg(self.model_name, dataset_dir=None)
            self.model = Net('LTLNet', self.model_name, 'test').to(self.device)

            # 猴子补丁：将 numpy._core 指向 numpy.core
            import numpy as np
            import sys
            if 'numpy._core' not in sys.modules:
                sys.modules['numpy._core'] = np.core
                # 可能还需要子模块
                sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath

            checkpoint = torch.load(self.pth_dir, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"模型加载失败: {e}")

    #初始化图像
    def init_process(self, img_path):
        img = Image.open(img_path).convert('I')
        orig_size = img.size  # (width, height) 原图尺寸
        img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized, dtype=np.float32)
        img_np = (img_np - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
        h, w = 256, 256
        img_pad = PadImg(img_np, times=32)
        img_tensor = torch.from_numpy(img_pad).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return img_resized, img_tensor, (h, w), orig_size

    #初始化视频/摄像头帧
    def init_frame(self,frame):
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        or_img = Image.fromarray(img).convert('L')
        orig_size = or_img.size
        img_resized = or_img.resize((256,256),Image.Resampling.LANCZOS)
        img_np = np.array(img_resized, dtype=np.float32)
        img_np = (img_np-self.img_norm_cfg['mean'])/self.img_norm_cfg['std']
        img_pad = PadImg(img_np,times=32)
        img_tensor = torch.from_numpy(img_pad).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return or_img,img_resized,img_tensor,(256,256),orig_size

    #获取背景
    def get_background(self, img, target_mask, find_field=4):
        h, w = (256,256)
        filled_img = img.copy()
        bg_coords = np.argwhere(~target_mask)
        if len(bg_coords) == 0:
            return filled_img

        target_coords = np.argwhere(target_mask)
        for (y, x) in target_coords:
            y_min = max(0, y - find_field)
            y_max = min(h, y + find_field)
            x_min = max(0, x - find_field)
            x_max = min(w, x + find_field)

            local_bg = bg_coords[
                (bg_coords[:, 0] >= y_min) & (bg_coords[:, 0] <= y_max) &
                (bg_coords[:, 1] >= x_min) & (bg_coords[:, 1] <= x_max)
                ]

            if len(local_bg) > 0:
                rand_y, rand_x = local_bg[np.random.choice(len(local_bg))]
                filled_img[y, x] = img[rand_y, rand_x]
            else:
                filled_img[y, x] = np.mean(img[bg_coords[:, 0], bg_coords[:, 1]])
        return filled_img

    #推理执行函数
    def run(self):
        self.load_model()
        if self.run_type == 0:
            total_model_time = 0.0
            total_frames = 0
            for idx, path in enumerate(self.paths):
                self.progress_signal.emit(idx + 1)
                img, img_tensor, original_size, orig_size = self.init_process(path)
                if img_tensor is None:
                    continue
                original_np = np.array(img, dtype=np.float32)
                h, w = original_size
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        output = self.model(img_tensor)
                        torch.cuda.synchronize()
                        total_frames += 1
                        output = output[:, :, :h, :w]
                        output_np = self.to_uint8(output)
                        pred_prob = output[0, 0, :, :].cpu().numpy()
                        target_mask = (pred_prob > 0.5)
                        y, x = np.where(target_mask)
                        field_size = 2
                        for target_y, target_x in zip(y, x):
                            y1 = max(0, target_y - field_size)
                            y2 = min(h, target_y + field_size)
                            x1 = max(0, target_x - field_size)
                            x2 = min(w, target_x + field_size)
                            target_mask[y1:y2, x1:x2] = True
                        background_np = self.get_background(original_np, target_mask)
                        background_np = (background_np - background_np.min()) / (background_np.max() - background_np.min() + 1e-8) * 255
                        background_np = background_np.astype(np.uint8)
                        output_img = Image.fromarray(output_np).resize(orig_size, Image.Resampling.NEAREST)
                        background_img = Image.fromarray(background_np).resize(orig_size, Image.Resampling.LANCZOS)
                        item_name = path.split('/')[-1].split('.')[0]
                        item_postfixs = path.split('/')[-1].split('.')[1]
                        save_path = os.path.join(self.result_dir, item_name)
                        os.makedirs(save_path, exist_ok=True)
                        output_path = os.path.join(save_path, f'pred.{item_postfixs}')
                        output_L_path = os.path.join(save_path, f'L.{item_postfixs}')
                        output_img.save(output_path)
                        background_img.save(output_L_path)

                        binary = (pred_prob > 0.5).astype(np.uint8)
                        labels = measure.label(binary, connectivity=2)
                        props = measure.regionprops(labels)
                        orig_rgb = Image.open(path).convert('RGB')
                        scale_w = orig_size[0] / 256.0
                        scale_h = orig_size[1] / 256.0
                        draw = ImageDraw.Draw(orig_rgb)
                        for prop in props:
                            y1, x1, y2, x2 = prop.bbox
                            x1 = int(x1 * scale_w)
                            y1 = int(y1 * scale_h)
                            x2 = int(x2 * scale_w)
                            y2 = int(y2 * scale_h)
                            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                        bbox_path = os.path.join(save_path, f'bbox.{item_postfixs}')
                        orig_rgb.save(bbox_path)
        elif self.run_type == 1:
            fps = 20
            fourcc = cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')
            for idx,path in enumerate(self.paths):
                fps_count = 0
                self.video = cv2.VideoCapture(path)
                self.fps_count_signal.emit(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
                item_name = path.split('/')[-1].split('.')[0]
                item_postfixs = path.split('/')[-1].split('.')[1]
                save_path = os.path.join(self.result_video_dir, item_name)
                os.makedirs(save_path, exist_ok=True)
                bbox_path = os.path.join(save_path, f'bbox.{item_postfixs}')
                ow = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
                oh = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_writer = cv2.VideoWriter(bbox_path, fourcc, fps, (ow, oh))
                if self.video.isOpened():
                    while True:
                        ret,frame = self.video.read()
                        self.progress_signal.emit(fps_count)
                        fps_count = fps_count + 1
                        if not ret:
                            break
                        or_img, img,img_tensor,original_size,orig_size =  self.init_frame(frame)
                        if img_tensor is None:
                            continue
                        h, w = original_size
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                output = self.model(img_tensor)
                                output = output[:, :, :h, :w]
                                pred_prob = output[0, 0, :, :].cpu().numpy()
                                binary = (pred_prob > 0.5).astype(np.uint8)
                                labels = measure.label(binary, connectivity=2)
                                props = measure.regionprops(labels)
                                orig_rgb = or_img.convert('RGB')
                                scale_w = orig_size[0] / 256.0
                                scale_h = orig_size[1] / 256.0
                                draw = ImageDraw.Draw(orig_rgb)
                                for prop in props:
                                    y1, x1, y2, x2 = prop.bbox
                                    x1 = int(x1 * scale_w)
                                    y1 = int(y1 * scale_h)
                                    x2 = int(x2 * scale_w)
                                    y2 = int(y2 * scale_h)
                                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                                frame_np = np.array(orig_rgb)
                                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                                self.video_writer.write(frame_bgr)
                self.video_writer.release()
        else:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                self.error_report.emit("无法打开摄像头")
                return
            while self.is_running:
                ret,frame = camera.read()
                if not ret:
                    self.error_report.emit("读取摄像头失败")
                    break
                or_img, img,img_tensor,original_size,orig_size =  self.init_frame(frame)
                h,w = original_size
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        output = output[:, :, :h, :w]
                        pred_prob = output[0, 0, :, :].cpu().numpy()

                        binary = (pred_prob > 0.5).astype(np.uint8)
                        labels = measure.label(binary, connectivity=2)
                        props = measure.regionprops(labels)
                        orig_rgb = or_img.convert('RGB')
                        scale_w = orig_size[0] / 256.0
                        scale_h = orig_size[1] / 256.0
                        draw = ImageDraw.Draw(orig_rgb)
                        for prop in props:
                            y1, x1, y2, x2 = prop.bbox
                            x1 = int(x1 * scale_w)
                            y1 = int(y1 * scale_h)
                            x2 = int(x2 * scale_w)
                            y2 = int(y2 * scale_h)
                            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                        result_np = np.array(orig_rgb)
                        h,w,c = result_np.shape
                        return_image = QImage(result_np.data,w,h,w*3,QImage.Format.Format_RGB888)
                        self.image_return.emit(return_image)
            camera.release()
        self.finished_signal.emit()

    def to_uint8(self, tensor):
        return (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    def stop_running(self):
        self.is_running = False
