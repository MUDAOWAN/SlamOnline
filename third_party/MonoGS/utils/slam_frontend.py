import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.online_semantic = self.config.get("OnlineSemantic", {})
        self.online_semantic_enabled = bool(self.online_semantic.get("enabled", False))
        self.online_semantic_queue_root = (
            Path(self.online_semantic["queue_root"])
            if self.online_semantic_enabled and self.online_semantic.get("queue_root")
            else None
        )
        self.online_semantic_overwrite = bool(self.online_semantic.get("overwrite", False))
        self.online_semantic_emit_init = bool(self.online_semantic.get("emit_init", True))
        self.online_semantic_emit_keyframes = bool(self.online_semantic.get("emit_keyframes", True))
        self.online_semantic_keyframe_stride = max(
            1, int(self.online_semantic.get("keyframe_stride", 1))
        )
        self.online_semantic_max_pending = int(self.online_semantic.get("max_pending", 0))
        self.online_semantic_dataset_root = Path(
            str(self.config["Dataset"]["dataset_path"])
        ).expanduser()
        self.online_semantic_seen_keyframes = 0

    def _ensure_online_semantic_queue_dirs(self):
        if self.online_semantic_queue_root is None:
            return
        for name in ("pending", "processing", "done", "failed", "adapter_runs"):
            (self.online_semantic_queue_root / name).mkdir(parents=True, exist_ok=True)

    def _write_online_semantic_json_atomic(self, path, data):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)

    def _semantic_frame_id_and_stem(self, dataset_idx):
        frame_stem = f"frame_{int(dataset_idx):06d}"
        frame_id = int(dataset_idx)
        color_paths = getattr(self.dataset, "color_paths", None)
        if color_paths is not None and dataset_idx < len(color_paths):
            path_stem = Path(str(color_paths[dataset_idx])).stem
            match = re.search(r"(\d+)$", path_stem)
            if match:
                frame_id = int(match.group(1))
                frame_stem = f"frame_{frame_id:06d}"
            else:
                frame_stem = path_stem
        return frame_id, frame_stem

    def _semantic_task_filename(self, task):
        task_index = int(task.get("task_index", task.get("trajectory_index", 0)))
        frame_id = int(task["frame_id"])
        return f"task_{task_index:06d}_frame_{frame_id:06d}.json"

    def _semantic_existing_paths(self, filename):
        return [
            self.online_semantic_queue_root / "pending" / filename,
            self.online_semantic_queue_root / "processing" / filename,
            self.online_semantic_queue_root / "done" / filename,
            self.online_semantic_queue_root / "failed" / filename,
        ]

    def _online_semantic_backlog(self):
        if self.online_semantic_queue_root is None:
            return 0
        pending = len(list((self.online_semantic_queue_root / "pending").glob("*.json")))
        processing = len(list((self.online_semantic_queue_root / "processing").glob("*.json")))
        return pending + processing

    def _camera_to_world_list(self, viewpoint):
        world_to_camera = getWorld2View2(viewpoint.R, viewpoint.T)
        camera_to_world = torch.linalg.inv(world_to_camera)
        return camera_to_world.detach().cpu().numpy().astype(float).tolist()

    def _path_relative_to_dataset(self, path):
        if path is None:
            return None
        try:
            return str(Path(str(path)).expanduser().resolve().relative_to(
                self.online_semantic_dataset_root.resolve()
            ))
        except ValueError:
            return None

    def _emit_online_semantic_task(self, cur_frame_idx, viewpoint, init=False):
        if not self.online_semantic_enabled or self.online_semantic_queue_root is None:
            return
        if init and not self.online_semantic_emit_init:
            return
        if (not init) and not self.online_semantic_emit_keyframes:
            return
        if not init:
            self.online_semantic_seen_keyframes += 1
            keyframe_ord = self.online_semantic_seen_keyframes - 1
            if keyframe_ord % self.online_semantic_keyframe_stride != 0:
                return

        try:
            self._ensure_online_semantic_queue_dirs()
            if (
                self.online_semantic_max_pending > 0
                and self._online_semantic_backlog() >= self.online_semantic_max_pending
            ):
                return
            frame_id, frame_stem = self._semantic_frame_id_and_stem(cur_frame_idx)
            color_paths = getattr(self.dataset, "color_paths", [])
            depth_paths = getattr(self.dataset, "depth_paths", [])
            image_path = str(color_paths[cur_frame_idx]) if cur_frame_idx < len(color_paths) else None
            depth_path = str(depth_paths[cur_frame_idx]) if cur_frame_idx < len(depth_paths) else None
            image_relative_path = self._path_relative_to_dataset(image_path)
            depth_relative_path = self._path_relative_to_dataset(depth_path)
            task = {
                "task_index": int(cur_frame_idx),
                "trajectory_index": int(cur_frame_idx),
                "monogs_frame_idx": int(cur_frame_idx),
                "frame_id": int(frame_id),
                "frame_stem": frame_stem,
                "is_init": bool(init),
                "is_keyframe": True,
                "dataset_root": str(self.online_semantic_dataset_root),
                "image_path": image_path,
                "image_relative_path": image_relative_path,
                "image_exists": bool(image_path and Path(image_path).exists()),
                "depth_path": depth_path,
                "depth_relative_path": depth_relative_path,
                "depth_exists": bool(depth_path and Path(depth_path).exists()),
                "timestamp": float(cur_frame_idx),
                "depth_timestamp": float(cur_frame_idx),
                "pose_c2w": self._camera_to_world_list(viewpoint),
                "intrinsics": {
                    "fx": float(viewpoint.fx),
                    "fy": float(viewpoint.fy),
                    "cx": float(viewpoint.cx),
                    "cy": float(viewpoint.cy),
                },
                "width": int(viewpoint.image_width),
                "height": int(viewpoint.image_height),
                "depth_scale": float(getattr(self.dataset, "depth_scale", 1.0) or 1.0),
            }

            filename = self._semantic_task_filename(task)
            pending_path = self.online_semantic_queue_root / "pending" / filename
            existing_paths = self._semantic_existing_paths(filename)
            if not self.online_semantic_overwrite and any(path.exists() for path in existing_paths):
                return
            if self.online_semantic_overwrite:
                for path in existing_paths:
                    if path.exists():
                        path.unlink()

            wrapper = {
                "format": "online_semantic_task_v1",
                "queue_status": "pending",
                "queued_at": datetime.now().isoformat(timespec="seconds"),
                "source": "monogs_frontend",
                "source_mode": "real_monogs_frontend_keyframe_callback",
                "event": {
                    "type": "init" if init else "keyframe",
                    "emitted_at": datetime.now().isoformat(timespec="seconds"),
                    "frame_id": int(frame_id),
                    "frame_stem": frame_stem,
                    "monogs_frame_idx": int(cur_frame_idx),
                    "is_init": bool(init),
                    "is_keyframe": True,
                },
                "task": task,
            }
            self._write_online_semantic_json_atomic(pending_path, wrapper)
            Log(f"Online semantic task queued: {pending_path}", tag="Semantic")
        except Exception as exc:
            Log(f"Online semantic task emit failed at frame {cur_frame_idx}: {exc}", tag="Semantic")

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1
        self._emit_online_semantic_task(cur_frame_idx, viewpoint, init=False)

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True
        self._emit_online_semantic_task(cur_frame_idx, viewpoint, init=True)

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        # eval_ate(
                        #     self.cameras,
                        #     self.kf_indices,
                        #     self.save_dir,
                        #     0,
                        #     final=True,
                        #     monocular=self.monocular,
                        # )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    # Log("Evaluating ATE at frame: ", cur_frame_idx)
                    # eval_ate(
                    #     self.cameras,
                    #     self.kf_indices,
                    #     self.save_dir,
                    #     cur_frame_idx,
                    #     monocular=self.monocular,
                    # )
                    pass
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
