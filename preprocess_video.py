from pathlib import Path
import argparse
import math
import cv2
from tqdm import tqdm
import numpy as np
import face_alignment
from utils import util

def preprocess_video(opt):
    out_dir = Path(opt.out_dir)
    util.mkdir(out_dir)

    device = "cuda" if opt.use_gpu else "cpu"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
    video = cv2.VideoCapture(opt.video)
    if not video.isOpened():
        raise NameError("Can not open the Video: {}".format(opt.video))

    fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Process video: {} with {:.1f} fps".format(opt.video, fps))

    frame_index = fps * opt.skip_head
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    print("Skip to frame: {}".format(frame_index))

    for i in tqdm(range(frame_index, n_frames - fps * opt.skip_tail), ascii=True):
        _, frame = video.read()
        if i % opt.skip_frame != 0:
            continue
        # TODO: crop region by input argument
        frame = frame[:, 500:-500, :]
        bbox = detect_face(fa, frame)
        if bbox is None:
            continue

        cropped = crop_face(frame, bbox, opt.crop_size)
        cv2.imwrite(str(out_dir / "{:0>4d}.png".format(i)), cropped)

    video.release()


def detect_face(fa, image):
    detected_faces = fa.face_detector.detect_from_image(image.copy())
    if detected_faces:
        area = [(face[2]-face[1])*(face[3]-face[0]) for face in detected_faces if face[4] > 0.7]
        max_idx = np.argmax(area)
        if max_idx < 0:
            return None
        bbox = detected_faces[max_idx][:-1].reshape(2, 2)
        return bbox
    else:
        return None

def crop_face(image, bbox, crop_size):
    center = bbox.mean(axis=0).astype(int)
    ox = max(center[0] - crop_size // 2, 0)
    oy = max(center[1] - crop_size // 2, 0)
    return image[oy:oy+crop_size, ox:ox+crop_size, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--skip_head", type=int, default=0)
    parser.add_argument("--skip_tail", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=512)

    opt, _ = parser.parse_known_args()
    preprocess_video(opt)
