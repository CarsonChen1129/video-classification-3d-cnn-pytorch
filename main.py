import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import glob

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video


# needs model, output, video_root
def extract_feats(opt, model):
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    dir_fc = opt.output_dir
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    video_list = glob.glob(os.path.join(opt.video_root,'*.mp4'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        outfile = os.path.join(dir_fc, video_id + '.npy')

        if os.path.exists(outfile):
            continue

        with open(os.devnull, "w") as ffmpeg_log:
            if os.path.exists(video):
                # print(video)
                subprocess.call('mkdir tmp', stdout=ffmpeg_log, stderr=ffmpeg_log,shell=True)
                subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video),
                                stdout=ffmpeg_log, stderr=ffmpeg_log, shell=True)

                result = classify_video('tmp', video, class_names, model, opt)
                # print(type(result))
                np.save(outfile, result)

                subprocess.call('rm -rf tmp', stdout=ffmpeg_log, stderr=ffmpeg_log,shell=True)
            else:
                print('{} does not exist'.format(video))




    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', stdout=ffmpeg_log, stderr=ffmpeg_log,shell=True)




if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)

    extract_feats(opt, model)




