import torch
import time
import math
import numpy as np
import dv_processing as dv
from spikingjelly.activation_based import functional
from .model_definition import SNN_Net, CONFIG
from src.shared import CameraObservation, CameraParams, default_camera_params
from dataclasses import dataclass, field
from .base import DVSLineAlgorithm

@dataclass
class SNNLineParams:
    cam_params: CameraParams = field(default_factory=default_camera_params)
    frame_decay: float = 0.0
    max_events: int = 0
    normalize_input: bool = False
    gaussian_sigma: float = 0.0
    clamp_output: bool = False
    max_slope: float = 0.0
    max_intercept: float = 0.0
    reset_each_step: bool = False
    
class SNN_regression(DVSLineAlgorithm):
    def __init__(self, params: SNNLineParams):
        cam = params.cam_params
        self.W = int(cam.DAVIS346_WIDTH)
        self.H = int(cam.DAVIS346_HEIGHT)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bModel = SNN_Net(
            tau=CONFIG["tau"],
            final_tau=CONFIG["final_tau"],
            hidden=CONFIG["hidden"],
            norm_type=CONFIG["norm_type"],
            learnable_norm=CONFIG["learnable_norm"],
            init_scale=CONFIG["init_scale"]
        )

        self.mModel = SNN_Net(
            tau=CONFIG["tau"],
            final_tau=CONFIG["final_tau"],
            hidden=CONFIG["hidden"],
            norm_type=CONFIG["norm_type"],
            learnable_norm=CONFIG["learnable_norm"],
            init_scale=CONFIG["init_scale"]
        )

        self.bModel.load_state_dict(torch.load(r"C:\Users\pcadm\Downloads\SNN\SNN-Regression-Pencil-Balancer-True\models\lin_b\model_SEW_BN\checkpoints_pendulum\best_model_weights.pth"))
        self.mModel.load_state_dict(torch.load(r"C:\Users\pcadm\Downloads\SNN\SNN-Regression-Pencil-Balancer-True\models\lin_m\model_SEW_BN\checkpoints_pendulum\best_model_weights.pth"))

        self.bModel.to(self.device)
        self.mModel.to(self.device)
        self.bModel.eval()
        self.mModel.eval()

        functional.reset_net(self.bModel)
        functional.reset_net(self.mModel)

        self.pos_frame = np.zeros((self.H, self.W), dtype=np.float32)
        self.neg_frame = np.zeros((self.H, self.W), dtype=np.float32)

    def update(self, events_np):

        if events_np is None or len(events_np) == 0:
            return None, None

        # 1. decay temporal (Hough memory)
        self.pos_frame *= self.frame_decay
        self.neg_frame *= self.frame_decay

        # 2. accumulate events
        xs = events_np['x']
        ys = events_np['y']
        ps = events_np['polarity']

        self.pos_frame[ys[ps == 1], xs[ps == 1]] += 1
        self.neg_frame[ys[ps == 0], xs[ps == 0]] += 1

        # 3. spatial smoothing (Hough inlier model)
        self.pos_frame = gaussian(self.pos_frame, sigma=self.sigma)
        self.neg_frame = gaussian(self.neg_frame, sigma=self.sigma)

        # 4. SNN inference
        frame = np.stack([self.pos_frame, self.neg_frame], axis=0)
        frame = torch.from_numpy(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            bOut = self.bModel(frame)
            mOut = self.mModel(frame)

        slope = (mOut.item() * 1.76e6) - 0.44e6
        intercept = (bOut.item() * 44000) - 27500

        # 5. safety (Hough determinant equivalent)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None, None

        slope = np.clip(slope, -1e6, 1e6)
        intercept = np.clip(intercept, -5e4, 5e4)

        return CameraObservation(slope, intercept)

    def reset(self):
        functional.reset_net(self.bModel)
        functional.reset_net(self.mModel)
        return CameraObservation(
            slope=0.0, 
            intercept= self.W / 2
        )
    
SNN_PRESETS = {
    "default": {
        # temporal smoothing ~ mixing_factor
        "frame_decay": 0.90,

        # event window control
        "max_events": 5000,

        # input normalization
        "normalize_input": True,

        # spatial smoothing ~ inlier_stddev_px
        "gaussian_sigma": 3.5,

        # safety ~ min_determinant
        "clamp_output": True,
        "max_slope": 1e6,
        "max_intercept": 5e4,

        # stability
        "reset_each_step": False,
    }
}