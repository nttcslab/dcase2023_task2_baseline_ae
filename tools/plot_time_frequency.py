import argparse, os, sys
from tools.plot_common import Figdata, show_figs

from datasets.datasets import Datasets

import numpy as np
import math
import torch

class TimeFrequencyFigData():
    def __init__(self,max_imgs=1, max_extract=1, frames=5, frame_hop_length=1, shape=(1,128,5)):
        self.img_count = [0, 0] #[nml_count, anm_count]
        self.figdatas = []
        self.max_imgs = max_imgs
        self.max_extract = max_extract
        self.anm_suffix = ["nml","anm"]
        self.frames = frames
        self.frame_hop_length = frame_hop_length
        self.shape = shape

    def append_figdata(self,data,label,machine_id,idx,fig_name="",is_fig_tern=False):
        if self.img_count[label] >= self.max_imgs:
            return
        imgs = data[ : self.frames*self.max_extract//self.frame_hop_length : self.frames//self.frame_hop_length].cpu()

        for i in range(len(imgs)):
            img = imgs[i]
            if is_fig_tern:
                img = torch.stack([img_tmp.T for img_tmp in img])
            img = img.view(self.shape)
            self.figdatas.append(Figdata(
                img.T,
                type="image",
                title = "ID{id}-{idx}_{anm}\n{frame_min}-{frame_max}frame\n{fig_name}".format(
                    id=machine_id,
                    idx=idx,
                    anm=self.anm_suffix[label],
                    frame_min=i*self.frames,
                    frame_max=(i+1)*self.frames,
                    fig_name=fig_name
                )
            ))
        self.img_count[label] += 1

    def show_fig(self, title="time_frequency", fold_interval=1, export_dir="results", is_display_console=False):
        show_figs(
            *self.figdatas,
            fold_interval=fold_interval,
            width_mm=50,
            margin_top_mm=30,
            margin_bottom_mm=30,
            margin_middle_mm=30,
            sup_title=title,
            export_path=f"{export_dir}/{title}.png",
            is_display_console=is_display_console
        )

    def reset_count(self):
        self.img_count = [0, 0]
