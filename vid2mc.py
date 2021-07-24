#!/usr/bin/env python
import os
import argparse
import pickle
import cv2
import numpy as np
from sklearn.neighbors import KDTree

block_path = os.path.join(os.path.dirname(__file__), 'block')
classifier_path = os.path.join(os.path.dirname(__file__), 'data.pkl')

# TODO: Update supported exts
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')
VIDEO_EXTS = ('.avi', '.mkv', '.mp4')


def generate_classifier(out_fname: str):
    """Generates block list
    """
    # Generate block list
    block_list = []
    for fname in os.listdir(block_path):
        if fname.endswith('.png'):
            im = cv2.imread(os.path.join(block_path, fname))
            h, w, _ = im.shape
            # Only whitelist 16x16 blocks
            if w == 16 and h == 16:
                avg_color = np.average(im, axis=(0, 1))
                block_list.append({
                    'path': os.path.join(block_path, fname),
                    'avg_color': avg_color,
                })

    # Generate KDTree
    X = [b['avg_color'] for b in block_list]
    lbls = [b['path'] for b in block_list]
    tree = KDTree(X)

    with open(out_fname, 'wb') as f:
        pickle.dump({
            'tree': tree,
            'lbls': lbls
        }, f)


def convert_im(im, tree, lbls, block_imgs):
    """Convert image to minecraft
    """
    h, w, _ = im.shape
    step = 16
    for r in range(0, h, step):
        for c in range(0, w, step):
            rnext = min(r+step, h)
            cnext = min(c+step, w)
            patch = im[r:rnext, c:cnext]
            color = np.average(patch, axis=(0, 1))

            # Get closest block
            _, ind = tree.query([color], k=1)
            lbl = lbls[ind[0][0]]
            block = block_imgs[lbl]

            # Copy values
            rmax = rnext-r
            cmax = cnext-c
            im[r:rnext, c:cnext] = block[:rmax, :cmax]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert videos and images to Minecraft.')
    parser.add_argument(
        'output', help='Output video/image file.')
    parser.add_argument(
        '-i', '--input', help='Input video/image file.', required=True)
    parser.add_argument(
        '--width', help='Output width.', type=int)
    parser.add_argument(
        '--height', help='Output height.', type=int)

    args = parser.parse_args()

    # Load classifier and labels
    if not os.path.exists(classifier_path):
        print('No classifier found. Generating classifier...')
        generate_classifier(classifier_path)

    print('Loading classifier...')
    with open(classifier_path, 'rb') as f:
        data = pickle.load(f)
        tree = data['tree']
        lbls = data['lbls']

    # Load block images
    print('Loading block images...')
    block_imgs = {}
    for lbl in lbls:
        block_imgs[lbl] = cv2.imread(lbl)

    in_fname = args.input
    out_fname = args.output
    w = args.width
    h = args.height

    # Check if in file exists
    if not os.path.exists(in_fname):
        print(f'Error: {in_fname} does not exist.')
        exit(1)

    if in_fname.endswith(IMAGE_EXTS):
        # Convert image
        print(f'Converting {in_fname}...')
        im = cv2.imread(in_fname)
        if not w:
            w = im.shape[1]
        if not h:
            h = im.shape[0]
        im = cv2.resize(im, (w, h))
        convert_im(im, tree, lbls, block_imgs)
        cv2.imwrite(out_fname, im)
        print(f'Saved to {out_fname}.')

    elif in_fname.endswith(VIDEO_EXTS):
        # Convert video
        cap = cv2.VideoCapture(in_fname)
        curr_frame = 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if cap.isOpened():
            if not w:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if not h:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_fname, fourcc, fps, (w, h))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (w, h))
            convert_im(frame, tree, lbls, block_imgs)
            out.write(frame)
            print(f'Converting frame {curr_frame}/{total_frames}...')
            curr_frame += 1
        cap.release()
        out.release()
        print(f'Saved to {out_fname}.')
