from tqdm import tqdm
import json
import codecs
import requests
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
from os import listdir
from os.path import isfile, join
import torch
import nltk
import numpy as np
import random
import re

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def preprocess_trafficqa_to_tensor_dataset():
    import clip
    import torch

    import pickle

    image_dir = 'data/frames/'

    train_metadata_dir = 'data/annotations/R3_train.json'
    val_metadata_dir = 'data/annotations/R3_test.json'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-L/16", device=device)

    torch.random.manual_seed(0)

    def read_image_and_qas(metadata_dir, split='train'):
        metadata = json_load(metadata_dir)

        all_images, all_questions, all_option_1, all_option_2, all_option_3, all_option_4, all_ans = [], [], [], [], [], [], []
        for ind, md in enumerate(tqdm(metadata[1:])):
            '''
            Abandoned due to significant use of memory
            '''
            # all_frames = []
            # frame_index = sorted(draw_samples([i for i in range(20)], 10))
            # for ind in frame_index:
            #     frame = preprocess(Image.open('{}{}_{}.jpg'.format(image_dir, key, str(ind))))
            #     all_frames.append(frame)
            # all_frames = torch.cat(all_frames, dim=0)
            all_frames = torch.tensor(md[1], dtype=torch.int).unsqueeze(0)  # video_id

            question = md[4]
            opt1 = md[5]
            opt2 = md[6]
            opt3 = md[7]
            opt4 = md[8]
            ans = torch.tensor(md[9], dtype=torch.long).unsqueeze(0)  # answer: option index

            t_question = clip.tokenize(question, context_length=77, truncate=True)
            # t_opt1 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt1), context_length=77, truncate=True)
            # t_opt2 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt2), context_length=77, truncate=True)
            # t_opt3 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt3), context_length=77, truncate=True)
            # t_opt4 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt4), context_length=77, truncate=True)

            t_opt1 = clip.tokenize('Answer: {}'.format(opt1), context_length=77, truncate=True)
            t_opt2 = clip.tokenize('Answer: {}'.format(opt2), context_length=77, truncate=True)
            t_opt3 = clip.tokenize('Answer: {}'.format(opt3), context_length=77, truncate=True)
            t_opt4 = clip.tokenize('Answer: {}'.format(opt4), context_length=77, truncate=True)

            all_images.append(all_frames)
            all_questions.append(t_question)
            all_option_1.append(t_opt1)
            all_option_2.append(t_opt2)
            all_option_3.append(t_opt3)
            all_option_4.append(t_opt4)
            all_ans.append(ans)

        pickle.dump(
            [all_images, all_questions, all_option_1, all_option_2, all_option_3, all_option_4, all_ans],
            open('data/{}_only_option.cache'.format(split), "wb"), protocol=4)

    read_image_and_qas(train_metadata_dir, split='train')
    read_image_and_qas(val_metadata_dir, split='val')


def sample_frames_from_video_trafficqa():
    # Importing all necessary libraries
    import cv2

    path = 'data/compressed_videos/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    video_names = json_load("data/annotations/vid_filename_to_id.json")

    frames_per_video = 25
    for ind, f in enumerate(tqdm(onlyfiles)):
        # Read the video from specified path
        cam = cv2.VideoCapture(path + f)

        # frame
        currentframe = 0
        all_frames = []
        while True:
            # reading from frame
            ret, frame = cam.read()

            if ret:
                all_frames.append(frame)
                currentframe += 1
            else:
                break
        lens = len(all_frames)
        if lens >= frames_per_video:
            interval = lens // frames_per_video

            frame_ind = [i * interval for i in range(frames_per_video)]
            for i in range(len(frame_ind)):
                if frame_ind[i] >= lens:
                    frame_ind[i] = lens - 1
            sampled_frames = [all_frames[i] for i in frame_ind]
        else:
            sampled_frames = sorted(draw_samples([i for i in range(len(all_frames))], frames_per_video))
            sampled_frames = [all_frames[i] for i in sampled_frames]

        for ind, frame in enumerate(sampled_frames):
            cv2.imwrite('data/frames/{}_{}.jpg'.format(f.split('.')[0], str(ind)), frame)

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    import math
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def reformat_jsonl_to_json():
    dir = 'data/annotations/R2_train.jsonl'

    with open(dir, 'r') as f:
        lines = f.readlines()

    new_js = []
    for line in lines:
        js = json.loads(line)
        new_js.append(js)

    json_dump(new_js, 'data/annotations/R2_train.json')


def resize_images():
    from PIL import Image

    path = 'data/frames/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for f in tqdm(onlyfiles):
        image = Image.open(path + f)
        image.thumbnail((224, 224))
        image.save(path.replace('frames', 'frames_resize') + f)


if __name__ == '__main__':
    sample_frames_from_video_trafficqa()
    preprocess_trafficqa_to_tensor_dataset()