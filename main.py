import logging
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.functional import F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import trange

from deepmag import dataset
from deepmag.model import MagNet
from deepmag.train import train_epoch


def train(dataset_root_dir, model_output_dir, *, num_epochs=3, batch_size=4,
          device="cpu", regularization_weight=0.1, learning_rate=0.0001,
          skip_epochs=0, load_model_path=None):
    device = torch.device(device)
    ds = dataset.from_dir(dataset_root_dir)
    if load_model_path:
        model = torch.load(load_model_path).to(device)
        logging.info("Loaded model from %s", load_model_path)
    else:
        model = MagNet().to(device)
    with trange(skip_epochs, num_epochs, 1, desc="Epoch") as pbar:
        for epoch_idx in pbar:
            train_epoch(model, ds, device, learning_rate=learning_rate,
                        batch_size=batch_size, reg_weight=regularization_weight)
            save_path = os.path.join(model_output_dir,
                                     '%s-b%s-r%s-lr%s-%02d.pt' % (time.strftime("%Y%m%d"), batch_size,
                                                                  regularization_weight, learning_rate,
                                                                  epoch_idx))
            torch.save(model, save_path)
            pbar.write("Saved snapshot to %s" % save_path)


def amplify(model_path, input_video, *, amplification=1.0, device="cpu", skip_frames=1):
    device = "cpu"
    #device = torch.device(device)
    model = torch.load(model_path, map_location=torch.device('cpu')).to(device)

    _to_tensor = transforms.ToTensor()
    cap = cv2.VideoCapture(input_video)

    # 準備輸出影片
    output_path = _video_output_path(input_video, amplification)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    last_frames = []
    num_skipped_frames = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = _to_tensor(pil_frame).to(device).unsqueeze(0)

        if len(last_frames) < num_skipped_frames:
            last_frames.append(frame_tensor)
            out.write(frame)
            continue

        amp_f_tensor = torch.tensor([[float(amplification)]], dtype=torch.float, device=device)
        pred_frame, _, _ = model.forward(last_frames[0], frame_tensor, amp_f_tensor)
        pred_frame = to_pil_image(pred_frame.squeeze(0).clamp(0, 1).detach().cpu())

        # 轉換成 OpenCV 格式並寫入影片
        pred_frame_cv = cv2.cvtColor(np.array(pred_frame), cv2.COLOR_RGB2BGR)
        out.write(pred_frame_cv)
        last_frames.append(frame_tensor)
        last_frames = last_frames[-num_skipped_frames:]

    cap.release()
    out.release()
    logging.info(f"Amplified video saved to: {output_path}")
    return output_path


def collage(output_video, *input_videos):
    clips = [cv2.VideoCapture(path) for path in input_videos]
    fps = clips[0].get(cv2.CAP_PROP_FPS)
    width = int(clips[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(clips[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_videos = len(input_videos)
    output_width = width * (2 if num_videos > 1 else 1)
    output_height = height * (2 if num_videos > 2 else 1)
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))
    font = ImageFont.truetype("arial.ttf", 32)

    while True:
        frames = []
        for clip in clips:
            ret, frame = clip.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Video {clips.index(clip)+1}", fill="white", font=font)
            frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        if len(frames) < num_videos:
            break

        # 組合影格
        collage_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        collage_frame[:height, :width] = frames[0]
        if num_videos > 1:
            collage_frame[:height, width:] = frames[1]
        if num_videos > 2:
            collage_frame[height:, :width] = frames[2]
        if num_videos > 3:
            collage_frame[height:, width:] = frames[3]

        out.write(collage_frame)

    for clip in clips:
        clip.release()
    out.release()
    logging.info(f"Collage video saved to: {output_video}")
    return output_video


def _video_output_path(input_path, amp_f):
    output_dir = os.path.dirname(input_path)
    output_basename, output_ext = os.path.splitext(os.path.basename(input_path))
    output_basename += '@{}x'.format(amp_f)
    output_path = os.path.join(output_dir, output_basename + output_ext)
    return output_path


if __name__ == "__main__":
    start = time.time()
    logging.basicConfig(level=logging.INFO)
    import clize
    clize.run((train, amplify, collage))
    end = time.time()
    elapsed = end - start
    print(f'Time taken: {elapsed:.6f} seconds')
