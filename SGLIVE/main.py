import pydiffvg
import torch
import cv2
import matplotlib.pyplot as plt
import random
import argparse
import math
import errno
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.nn.functional import adaptive_avg_pool2d
from torchmetrics.image import PeakSignalNoiseRatio
import warnings
warnings.filterwarnings("ignore")

import PIL
import PIL.Image
import os
import os.path as osp
import numpy as np
import numpy.random as npr
import shutil
import copy
from xing_loss import xing_loss

import yaml
from easydict import EasyDict as edict

from skimage.filters import correlate_sparse, threshold_otsu, threshold_multiotsu
from skimage.morphology import binary_dilation, binary_erosion, convex_hull_image
from skimage.segmentation import watershed, expand_labels
import skimage.io

pydiffvg.set_print_timing(False)
gamma = 1.0

##########
# helper #
##########

from utils import \
    get_experiment_id, \
    get_path_schedule, \
    edict_2_dict, \
    check_and_create_dir

def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments*3)
    for i in range(0, segments*3):
        point = (np.cos(np.deg2rad(i * avg_degree)),
                    np.sin(np.deg2rad(i * avg_degree)))
        points.append(point)
    points = torch.tensor(points)
    points = (points)*radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points

def get_sdf(phi, method='skfmm', **kwargs):
    if method == 'skfmm':
        import skfmm
        phi = (phi-0.5)*2
        if (phi.max() <= 0) or (phi.min() >= 0):
            return np.zeros(phi.shape).astype(np.float32)
        sd = skfmm.distance(phi, dx=1)

        flip_negative = kwargs.get('flip_negative', True)
        if flip_negative:
            sd = np.abs(sd)

        truncate = kwargs.get('truncate', 10)
        sd = np.clip(sd, -truncate, truncate)
        # print(f"max sd value is: {sd.max()}")

        zero2max = kwargs.get('zero2max', True)
        if zero2max and flip_negative:
            sd = sd.max() - sd
        elif zero2max:
            raise ValueError

        normalize = kwargs.get('normalize', 'sum')
        if normalize == 'sum':
            sd /= sd.sum()
        elif normalize == 'to1':
            sd /= sd.max()
        return sd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--config", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target", type=str, help="target image path")
    parser.add_argument('--log_dir', metavar='DIR', default="log/debug")
    parser.add_argument('--initial', type=str, default="random", choices=['random', 'circle'])
    parser.add_argument('--signature', nargs='+', type=str)
    parser.add_argument('--seginit', nargs='+', type=str)
    parser.add_argument("--num_segments", type=int, default=4)
    parser.add_argument("--use_gradient", action='store_true', default=None)
    parser.add_argument("--init_opacity", type=float)
    parser.add_argument("--background", type=str, choices=['auto', 'yes', 'no'], default='auto')
    cfg = edict()
    args = parser.parse_args()
    cfg.debug = args.debug
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.target = args.target
    cfg.log_dir = args.log_dir
    cfg.initial = args.initial
    cfg.signature = args.signature
    # set cfg num_segments in command
    cfg.num_segments = args.num_segments
    if args.use_gradient is not None:
        cfg.use_gradient = args.use_gradient
    if args.background is not None:
        cfg.background = args.background
    if args.init_opacity is not None:
        cfg.init_opacity = args.init_opacity
    if args.seginit is not None:
        cfg.seginit = edict()
        cfg.seginit.type = args.seginit[0]
        if cfg.seginit.type == 'circle':
            cfg.seginit.radius = float(args.seginit[1])
    return cfg

class random_coord_init():
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size
    def __call__(self):
        h, w = self.canvas_size
        return [npr.uniform(0, 1)*w, npr.uniform(0, 1)*h]

class naive_coord_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]', replace_sampling=True):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0])**2).sum(0)
        elif format == ['[2D x c]']:
            self.map = ((pred - gt)**2).sum(-1)
        else:
            raise ValueError
        self.replace_sampling = replace_sampling

    def __call__(self):
        coord = np.where(self.map == self.map.max())
        coord_h, coord_w = coord[0][0], coord[1][0]
        if self.replace_sampling:
            self.map[coord_h, coord_w] = -1
        return [coord_w, coord_h]


class sparse_coord_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]', quantile_interval=200, nodiff_thres=0.1):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if format == '[bs x c x 2D]':
            self.map = ((pred[0] - gt[0])**2).sum(0)
            self.reference_gt = copy.deepcopy(
                np.transpose(gt[0], (1, 2, 0)))
        elif format == ['[2D x c]']:
            self.map = (np.abs(pred - gt)).sum(-1)
            self.reference_gt = copy.deepcopy(gt[0])
        else:
            raise ValueError
        # OptionA: Zero too small errors to avoid the error too small deadloop
        self.map[self.map < nodiff_thres] = 0
        quantile_interval = np.linspace(0., 1., quantile_interval)
        quantized_interval = np.quantile(self.map, quantile_interval)
        # remove redundant
        quantized_interval = np.unique(quantized_interval)
        quantized_interval = sorted(quantized_interval[1:-1])
        self.map = np.digitize(self.map, quantized_interval, right=False)
        self.map = np.clip(self.map, 0, 255).astype(np.uint8)
        self.idcnt = {}
        for idi in sorted(np.unique(self.map)):
            self.idcnt[idi] = (self.map==idi).sum()
        self.idcnt.pop(min(self.idcnt.keys()))
        # remove smallest one to remove the correct region
    def __call__(self):
        if len(self.idcnt) == 0:
            h, w = self.map.shape
            return [npr.uniform(0, 1)*w, npr.uniform(0, 1)*h]
        target_id = max(self.idcnt, key=self.idcnt.get)
        _, component, cstats, ccenter = cv2.connectedComponentsWithStats(
            (self.map==target_id).astype(np.uint8), connectivity=4)
        # remove cid = 0, it is the invalid area
        csize = [ci[-1] for ci in cstats[1:]]
        target_cid = csize.index(max(csize))+1
        center = ccenter[target_cid][::-1]
        coord = np.stack(np.where(component == target_cid)).T
        dist = np.linalg.norm(coord-center, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]
        # replace_sampling
        self.idcnt[target_id] -= max(csize)
        if self.idcnt[target_id] == 0:
            self.idcnt.pop(target_id)
        self.map[component == target_cid] = 0
        return [coord_w, coord_h]


class segmentation_guided_coord_init():
    def __init__(self, pred, gt, format='[bs x c x 2D]', nodiff_thres=0.1):
        if format == '[bs x c x 2D]':
            self.diff = (pred[0] - gt[0]).detach().cpu().numpy()
            self.diff_l2 = (self.diff**2).sum(0)
        else:
            raise ValueError
        nodiff_otsu = threshold_otsu(self.diff_l2)
        nodiff_thres = np.clip(nodiff_otsu, 0.04, nodiff_thres)
        self.nodiff_thres = nodiff_thres
        self.diff[(self.diff_l2 < nodiff_thres)[None].repeat(3,axis=0)] = 0
        laplacian_filter = np.array(
            [[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)[None, :, :]
        diff_lap = correlate_sparse(
            self.diff, laplacian_filter, mode='reflect')
        diff_lap = np.sum(np.abs(diff_lap), axis=0)
        diff_lap = diff_lap / max(diff_lap.max(), 1.0)
        otsu_threshold = threshold_multiotsu(diff_lap)[0]# / 2
        diff_lap = np.where(diff_lap < otsu_threshold, 0, 1)
        selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        dil = binary_dilation(diff_lap, footprint=selem).astype(np.int32)
        ero = binary_erosion(dil, footprint=selem).astype(np.int32)
        ws = watershed(ero, markers=None, connectivity=4)
        self.seg = expand_labels(ws, distance=2)
        self.weight = {}
        self.centroid = {}
        self.queried_segs = np.zeros_like(self.seg)
        diffl2_mask = self.diff_l2 >= nodiff_thres
        seg_keys = sorted(np.unique(self.seg))
        bg_sid = seg_keys[0]
        for idi in seg_keys[1:]:
            mask = self.seg == idi
            area = mask.sum()
            weight = self.diff_l2[mask & diffl2_mask].sum()
            if weight / area < nodiff_thres:
                continue
            self.weight[idi] = weight
            self.centroid[idi] = (np.argwhere(self.seg == idi)).mean(axis=0)
        self.bg = self.seg == bg_sid

    def __call__(self):
        if len(self.weight) == 0:
            p = np.clip(np.sum(self.diff, axis=0) + 1e-2, 0, 1)
            if p.sum() < 1e-3:
                p = np.ones_like(p)
            p /= p.sum()
            ref = np.unravel_index(npr.choice(p.size, p=p.flatten()), p.shape)
            self.queried_segs = np.ones_like(self.seg)
            return [ref[1], ref[0]]
        seg_id = max(self.weight, key=self.weight.get)
        seg_centroid = self.centroid[seg_id]
        coord = np.stack(np.where(self.seg == seg_id)).T
        dist = np.linalg.norm(coord - seg_centroid, axis=1)
        target_coord_id = np.argmin(dist)
        coord_h, coord_w = coord[target_coord_id]
        self.weight.pop(seg_id)
        self.last_seg = self.seg == seg_id
        self.queried_segs += self.last_seg
        return [coord_w, coord_h]

def init_shapes(num_paths,
                num_segments,
                canvas_size,
                seginit_cfg,
                shape_cnt,
                pos_init_method=None,
                trainable_stroke=False,
                use_gradient=False,
                init_opacity=1.0,
                **kwargs):
    shapes = []
    shape_groups = []
    h, w = canvas_size

    # change path init location
    if pos_init_method is None:
        pos_init_method = random_coord_init(canvas_size=canvas_size)

    for i in range(num_paths):
        num_control_points = [2] * num_segments

        if seginit_cfg.type=="random":
            points = []
            p0 = pos_init_method()
            color_ref = copy.deepcopy(p0)
            points.append(p0)
            for j in range(num_segments):
                radius = seginit_cfg.radius
                p1 = (p0[0] + radius * npr.uniform(-0.5, 0.5),
                      p0[1] + radius * npr.uniform(-0.5, 0.5))
                p2 = (p1[0] + radius * npr.uniform(-0.5, 0.5),
                      p1[1] + radius * npr.uniform(-0.5, 0.5))
                p3 = (p2[0] + radius * npr.uniform(-0.5, 0.5),
                      p2[1] + radius * npr.uniform(-0.5, 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.FloatTensor(points)

        # circle points initialization
        elif seginit_cfg.type=="circle":
            radius = seginit_cfg.radius
            if radius is None:
                radius = npr.uniform(0.5, 1)
            center = pos_init_method()
            color_ref = copy.deepcopy(center)
            points = get_bezier_circle(
                radius=radius, segments=num_segments,
                bias=center)

        path = pydiffvg.Path(num_control_points = torch.LongTensor(num_control_points),
                             points = points,
                             stroke_width = torch.tensor(0.0),
                             is_closed = True)
        shapes.append(path)

        if 'gt' in kwargs:
            wref, href = color_ref
            wref = max(0, min(int(wref), w-1))
            href = max(0, min(int(href), h-1))
            reference_color = list(kwargs['gt'][0, :, href, wref]) + [init_opacity]
            reference_color = torch.FloatTensor(reference_color)
            stroke_color_init = torch.FloatTensor(npr.uniform(size=[4]))
            if use_gradient:
                canvas_size_np = np.array(canvas_size, dtype=np.float32)
                canvas_size = torch.FloatTensor(canvas_size_np).requires_grad_(False)
                gradient_radius_ref = 0.5, 0.5
                if hasattr(pos_init_method, 'last_seg'):
                    def bbox2(img):
                        rows = np.any(img, axis=1)
                        cols = np.any(img, axis=0)
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        return rmin, rmax, cmin, cmax
                    rmin, rmax, cmin, cmax = bbox2(pos_init_method.last_seg)
                    gradient_radius_ref = np.clip((rmax - rmin) / h / 2, 0.1, 0.5), np.clip((cmax - cmin) / w / 2, 0.1, 0.5)
                gradient_r = np.sqrt(gradient_radius_ref[0] * gradient_radius_ref[1])
                print(f"initial gradient radius: {gradient_r}")
                gradient_params = {
                    'center': torch.FloatTensor(np.array([wref, href]) / canvas_size_np) * canvas_size,
                    'radius': torch.FloatTensor([gradient_r, gradient_r]) * canvas_size,
                    'offsets': torch.FloatTensor([0.0, 1.0]),
                    'stop_colors': torch.stack([reference_color, reference_color]),
                }
                fill_color_init = pydiffvg.RadialGradient(
                    center = gradient_params['center'],
                    radius = gradient_params['radius'],
                    offsets = gradient_params['offsets'],
                    stop_colors = gradient_params['stop_colors'],
                )
                fill_color_params = list(gradient_params.values())
            else:
                fill_color_init = reference_color
                fill_color_params = [reference_color]
        else:
            fill_color_init = torch.FloatTensor(npr.uniform(size=[4]))
            stroke_color_init = torch.FloatTensor(npr.uniform(size=[4]))

        path_group = pydiffvg.ShapeGroup(
            shape_ids = torch.LongTensor([shape_cnt+i]),
            fill_color = fill_color_init,
            stroke_color = stroke_color_init,
        )
        path_group.fill_color_params = fill_color_params
        shape_groups.append(path_group)

    point_var = []
    color_var = []

    for path in shapes:
        path.points.requires_grad_(True)
        point_var.append(path.points)
    for group in shape_groups:
        for param in group.fill_color_params:
            param.requires_grad_(True)
        color_var += group.fill_color_params

    if trainable_stroke:
        stroke_width_var = []
        stroke_color_var = []
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_var.append(path.stroke_width)
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            stroke_color_var.append(group.stroke_color)
        return shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var
    else:
        return shapes, shape_groups, point_var, color_var

class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n//self.decay_every
        decay_step = n %self.decay_every
        lr_s = self.decay_ratio**decay_time
        lr_e = self.decay_ratio**(decay_time+1)
        r = decay_step/self.decay_every
        lr = lr_s * (1-r) + lr_e * r
        return lr


def main():
    ###############
    # make config #
    ###############

    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_default = edict(cfg['default'])
    cfg = edict(cfg[cfg_arg.experiment])
    cfg.update(cfg_default)
    cfg.update(cfg_arg)
    cfg.exid = get_experiment_id(cfg.debug)

    cfg.experiment_dir = \
        osp.join(cfg.log_dir, '{}_{}'.format(cfg.exid, '_'.join(cfg.signature)))
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    gt = skimage.io.imread(cfg.target)
    alpha = None
    print(f"Input image shape is: {gt.shape}")
    if len(gt.shape) == 2:
        print("Converting the gray-scale image to RGB.")
        gt = gt.unsqueeze(dim=-1).repeat(1,1,3)
        gt = (gt/255).astype(np.float32)
    elif gt.shape[2] == 3:
        gt = (gt/255).astype(np.float32)
    elif gt.shape[2] == 4:
        print("Input image includes alpha channel, blend it with white background")
        gt = gt[:, :, :4]
        gt = (gt/255).astype(np.float32)
        alpha = gt[..., 3]
        gt = gt[..., :3] * gt[..., 3:] + (1 - gt[..., 3:])
    gt = torch.FloatTensor(gt).permute(2, 0, 1)[None].to(device)
    h, w = gt.shape[2:]

    path_schedule = get_path_schedule(**cfg.path_schedule)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    render = pydiffvg.RenderFunction.apply

    shapes_record, shape_groups_record = [], []

    region_loss = None
    loss_matrix = []

    para_point, para_color = {}, {}
    if cfg.trainable.stroke:
        para_stroke_width, para_stroke_color = {}, {}

    pathn_record = []
    svg_background = None
    def to_background_str(rgb, alpha):
        return f"background-color: rgba({rgb[0] * 255.0}, {rgb[1] * 255.0}, {rgb[2] * 255.0}, {alpha})"
    # Background
    if cfg.coord_init.type == 'segmentation':
        seg_bg = segmentation_guided_coord_init(torch.FloatTensor([1, 1, 1]), gt).bg
        gt_linear = gt.permute(0,2,3,1)[0] ** 2.2
        color_bg = gt_linear[seg_bg].mean(dim=0) ** (1 / 2.2)
        if alpha is not None:
            mean_bg_alpha = alpha[seg_bg].mean()
        else:
            mean_bg_alpha = 1.0
        if cfg.background == 'auto':
            if mean_bg_alpha > 0.6:
                svg_background = to_background_str(color_bg, mean_bg_alpha)
        elif cfg.background == 'yes':
            svg_background = to_background_str(color_bg, 1.0)
    else:
        color_bg = torch.tensor([1., 1., 1.])
    if cfg.trainable.bg:
        para_bg = torch.tensor(color_bg, requires_grad=True, device=device)
    else:
        para_bg = color_bg.detach().clone().requires_grad_(False).to(device)

    ##################
    # start_training #
    ##################

    loss_weight = None
    distance_weight_keep = 0
    selected_pixels_keep = 0
    distance_weight = None
    selected_pixels = None
    if cfg.coord_init.type == 'naive':
        pos_init_method = naive_coord_init(
            para_bg.view(1, -1, 1, 1).repeat(1, 1, h, w), gt)
    elif cfg.coord_init.type == 'sparse':
        pos_init_method = sparse_coord_init(
            para_bg.view(1, -1, 1, 1).repeat(1, 1, h, w), gt)
    elif cfg.coord_init.type == 'segmentation':
        pos_init_method = segmentation_guided_coord_init(
            para_bg.view(1, -1, 1, 1).repeat(1, 1, h, w), gt)
    elif cfg.coord_init.type == 'random':
        pos_init_method = random_coord_init([h, w])
    else:
        raise ValueError

    lrlambda_f = linear_decay_lrlambda_f(cfg.num_iter, 0.4)
    optim_schedular_dict = {}

    metric_psnr = PeakSignalNoiseRatio().to(device)

    for path_idx, pathn in enumerate(path_schedule):
        loss_list = []
        print("=> Adding [{}] paths, [{}] ...".format(pathn, cfg.seginit.type))
        pathn_record.append(pathn)
        pathn_record_str = f"{sum(pathn_record)}_{'-'.join([str(i) for i in pathn_record])}"

        # initialize new shapes related stuffs.
        if cfg.trainable.stroke:
            shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var = init_shapes(
                pathn, cfg.num_segments, (h, w),
                cfg.seginit, len(shapes_record),
                pos_init_method,
                trainable_stroke=True,
                use_gradient=cfg.use_gradient,
                init_opacity=cfg.init_opacity,
                gt=gt, )
            para_stroke_width[path_idx] = stroke_width_var
            para_stroke_color[path_idx] = stroke_color_var
        else:
            shapes, shape_groups, point_var, color_var = init_shapes(
                pathn, cfg.num_segments, (h, w),
                cfg.seginit, len(shapes_record),
                pos_init_method,
                trainable_stroke=False,
                use_gradient=cfg.use_gradient,
                init_opacity=cfg.init_opacity,
                gt=gt, )

        shapes_record += shapes
        shape_groups_record += shape_groups

        if cfg.save.init:
            filename = os.path.join(
                cfg.experiment_dir, "svg-init",
                "{}-init.svg".format(pathn_record_str))
            check_and_create_dir(filename)
            pydiffvg.save_svg(
                filename, w, h,
                shapes_record, shape_groups_record,
                background=svg_background)

        para = {}
        if (cfg.trainable.bg) and (path_idx == 0):
            para['bg'] = [para_bg]
        para['point'] = point_var
        para['color'] = color_var
        if cfg.trainable.stroke:
            para['stroke_width'] = stroke_width_var
            para['stroke_color'] = stroke_color_var

        pg = [{'params' : para[ki], 'lr' : cfg.lr_base[ki]} for ki in sorted(para.keys())]
        optim = torch.optim.Adam(pg)

        if cfg.trainable.record:
            scheduler = LambdaLR(
                optim, lr_lambda=lrlambda_f, last_epoch=-1)
        else:
            scheduler = LambdaLR(
                optim, lr_lambda=lrlambda_f, last_epoch=cfg.num_iter)
        optim_schedular_dict[path_idx] = (optim, scheduler)

        # Inner loop training
        t_range = tqdm(range(cfg.num_iter))
        for t in t_range:

            for _, (optim, _) in optim_schedular_dict.items():
                optim.zero_grad()

            # Forward pass: render the image.
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_record, shape_groups_record)
            img = render(w, h, 2, 2, t, None, *scene_args)

            # Compose img with inferred background
            img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:])

            if cfg.save.video:
                filename = os.path.join(
                    cfg.experiment_dir, "video-png",
                    "{}-iter{}.png".format(pathn_record_str, t))
                check_and_create_dir(filename)
                imshow = img.detach().cpu()
                pydiffvg.imwrite(imshow, filename, gamma=gamma)

            x = img.unsqueeze(0).permute(0, 3, 1, 2) # HWC -> NCHW

            loss = ((x-gt)**2)

            if cfg.loss.use_l1_loss:
                loss = abs(x-gt)

            if cfg.loss.use_distance_weighted_loss or cfg.loss.use_segmentation_weight:
                shapes_forsdf = copy.deepcopy(shapes)
                shape_groups_forsdf = copy.deepcopy(shape_groups)
                for si in shapes_forsdf:
                    si.stroke_width = torch.FloatTensor([0]).to(device)
                for sg_idx, sgi in enumerate(shape_groups_forsdf):
                    sgi.fill_color = torch.FloatTensor([1, 1, 1, 1]).to(device)
                    sgi.shape_ids = torch.LongTensor([sg_idx]).to(device)

                sargs_forsdf = pydiffvg.RenderFunction.serialize_scene(
                    w, h, shapes_forsdf, shape_groups_forsdf)
                with torch.no_grad():
                    im_forsdf = render(w, h, 2, 2, 0, None, *sargs_forsdf)
                # use alpha channel is a trick to get 0-1 image
                im_forsdf = (im_forsdf[:, :, 3]).detach().cpu().numpy()

            if cfg.loss.use_distance_weighted_loss:
                distance_weight = get_sdf(im_forsdf, normalize='to1')
                loss_weight = np.clip(distance_weight + distance_weight_keep, 0.0, 1.0)

            if cfg.loss.use_segmentation_weight:
                # get segmentation ids
                if not cfg.coord_init.type == 'segmentation':
                    raise ValueError("Segmentation weight is only supported with segmentation guided init")
                seg_fill_portion = 0.6
                queried_segs = pos_init_method.queried_segs 
                # select pixels inside and queried pixels
                selected_pixels = np.where(im_forsdf * queried_segs >= 0.99, 1, 0)
                all_selected_pixels = np.clip(selected_pixels + selected_pixels_keep, 0.0, 1.0)
                seg_reweight = np.where(all_selected_pixels, 1, 1 - seg_fill_portion)
                if loss_weight is None:
                    loss_weight = np.zeros([h, w], dtype=np.float32)
                loss_weight = np.clip(loss_weight * seg_reweight, all_selected_pixels * seg_fill_portion, 1.0)

            loss_weight = np.clip(loss_weight, 0, 1)
            loss_weight = torch.FloatTensor(loss_weight).to(device)

            if cfg.save.loss:
                save_loss = loss.squeeze(dim=0).mean(dim=0,keepdim=False).cpu().detach().numpy()
                save_weight = loss_weight.cpu().detach().numpy()
                save_weighted_loss = save_loss*save_weight
                # normalize to [0,1]
                save_loss = (save_loss - np.min(save_loss))/np.ptp(save_loss)
                save_weight = (save_weight - np.min(save_weight))/np.ptp(save_weight)
                save_weighted_loss = (save_weighted_loss - np.min(save_weighted_loss))/np.ptp(save_weighted_loss)

                # save
                plt.imshow(save_loss, cmap='Reds')
                plt.axis('off')
                filename = os.path.join(cfg.experiment_dir, "loss", "{}-iter{}-mseloss.png".format(pathn_record_str, t))
                check_and_create_dir(filename)
                plt.savefig(filename, dpi=800)
                plt.close()

                plt.imshow(save_weight, cmap='Greys')
                plt.axis('off')
                filename = os.path.join(cfg.experiment_dir, "loss", "{}-iter{}-sdfweight.png".format(pathn_record_str, t))
                plt.savefig(filename, dpi=800)
                plt.close()

                plt.imshow(save_weighted_loss, cmap='Reds')
                plt.axis('off')
                filename = os.path.join(cfg.experiment_dir, "loss", "{}-iter{}-weightedloss.png".format(pathn_record_str, t))
                plt.savefig(filename, dpi=800)
                plt.close()


            if loss_weight is None:
                loss = loss.sum(1).mean()
            else:
                loss = (loss.sum(1)*loss_weight).mean()

            if (cfg.loss.xing_loss_weight is not None) \
                    and (cfg.loss.xing_loss_weight > 0):
                loss_xing = xing_loss(point_var) * cfg.loss.xing_loss_weight
                loss = loss + loss_xing


            loss_list.append(loss.item())
            t_range.set_postfix({'loss': loss.item()})
            loss.backward()

            # step
            for _, (optim, scheduler) in optim_schedular_dict.items():
                optim.step()
                scheduler.step()

            with torch.no_grad():
                for group in shape_groups_record:
                    if isinstance(group.fill_color, torch.Tensor):
                        group.fill_color.clamp_(0.0, 1.0)
                    else:
                        group.fill_color.offsets.clamp_(0.0, 1.0)
                        group.fill_color.stop_colors.clamp_(0.0, 1.0)

        if distance_weight is not None:
            distance_weight_keep = np.clip(distance_weight_keep + distance_weight, 0.0, 1.0)
        if selected_pixels is not None:
            selected_pixels_keep = np.clip(selected_pixels_keep + selected_pixels, 0.0, 1.0)

        if not cfg.trainable.record:
            for _, pi in pg.items():
                for ppi in pi:
                    pi.require_grad = False
            optim_schedular_dict = {}

        if cfg.save.image:
            filename = os.path.join(
                cfg.experiment_dir, "demo-png", "{}.png".format(pathn_record_str))
            check_and_create_dir(filename)
            imshow = img.detach().cpu()
            if loss_weight != None:
                pydiffvg.imwrite(imshow, filename, gamma=gamma)
                filename = os.path.join(cfg.experiment_dir, "debug-png", "{}-loss_weight.png".format(pathn_record_str))
                check_and_create_dir(filename)
                skimage.io.imsave(filename, loss_weight.cpu().detach().numpy())
                if cfg.loss.use_segmentation_weight:
                    filename = os.path.join(cfg.experiment_dir, "debug-png", "{}-selected_pixels_keep.png".format(pathn_record_str))
                    skimage.io.imsave(filename, selected_pixels_keep)
                if cfg.loss.use_distance_weighted_loss:
                    filename = os.path.join(cfg.experiment_dir, "debug-png", "{}-distance_weight_keep.png".format(pathn_record_str))
                    skimage.io.imsave(filename, distance_weight_keep)

        if cfg.save.output:
            filename = os.path.join(
                cfg.experiment_dir, "output-svg", "{}.svg".format(pathn_record_str))
            check_and_create_dir(filename)
            pydiffvg.save_svg(filename, w, h, shapes_record, shape_groups_record, background=svg_background)

        loss_matrix.append(loss_list)

        pos_init_method = naive_coord_init(x, gt)

        if cfg.coord_init.type == 'naive':
            pos_init_method = naive_coord_init(x, gt)
        elif cfg.coord_init.type == 'sparse':
            pos_init_method = sparse_coord_init(x, gt)
        elif cfg.coord_init.type == 'segmentation':
            pos_init_method = segmentation_guided_coord_init(x, gt)
        elif cfg.coord_init.type == 'random':
            pos_init_method = random_coord_init([h, w])
        else:
            raise ValueError

        if cfg.save.video:
            print("saving iteration video...")
            img_array = []
            for ii in range(0, cfg.num_iter):
                filename = os.path.join(
                    cfg.experiment_dir, "video-png", 
                    "{}-iter{}.png".format(pathn_record_str, ii))
                img = cv2.imread(filename)
                # cv2.putText(
                #     img, "Path:{} \nIteration:{}".format(pathn_record_str, ii), 
                #     (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                img_array.append(img)

            videoname = os.path.join(
                cfg.experiment_dir, "video-avi", 
                "{}.avi".format(pathn_record_str))
            check_and_create_dir(videoname)
            out = cv2.VideoWriter(
                videoname, 
                # cv2.VideoWriter_fourcc(*'mp4v'),
                cv2.VideoWriter_fourcc(*'FFV1'), 
                20.0, (w, h))
            for iii in range(len(img_array)):
                out.write(img_array[iii])
            out.release()
            # shutil.rmtree(os.path.join(cfg.experiment_dir, "video-png"))

        psnr = metric_psnr(x, gt)
        if cfg.debug:
            with open(os.path.join(cfg.experiment_dir, "log.txt"), 'a') as f:
                f.write(json.dumps({
                    "total_paths": sum(pathn_record),
                    "pathn_record": pathn_record,
                    "psnr": psnr.item(),
                }) + "\n")

    print("The last loss is: {}".format(loss.item()))


if __name__ == "__main__":
    main()
