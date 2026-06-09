import os
import h5py
import numpy as np

from natsort import natsorted

import torch
from torch.utils.data import Dataset

from typing import List, Dict

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.


NORM_EPS = 1e-8


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


class InfiniteFileGroupedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, n_files, patches_per_vol, batch_size,
                 shuffle_files=True, shuffle_within=True, seed=0, drop_last=True):
        self.n_files = int(n_files)
        self.ppv = int(patches_per_vol)
        self.bs = int(batch_size)
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_within = bool(shuffle_within)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        files = np.arange(self.n_files)

        while True:
            if self.shuffle_files:
                self.rng.shuffle(files)

            for f in files:
                base = f * self.ppv
                offs = np.arange(self.ppv)
                if self.shuffle_within:
                    self.rng.shuffle(offs)

                for i in range(0, self.ppv, self.bs):
                    batch = base + offs[i:i+self.bs]
                    if len(batch) < self.bs and self.drop_last:
                        continue
                    yield batch.tolist()

    def __len__(self):
        return 10**12


def random_crop_3d(arr, patch_size):
    if isinstance(patch_size, int):
        pd = ph = pw = patch_size
    else:
        pd, ph, pw = patch_size

    if arr.ndim == 3:
        D, H, W = arr.shape
        has_channel = False
    elif arr.ndim == 4:
        C, D, H, W = arr.shape
        has_channel = True
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")

    if not has_channel:
        D, H, W = arr.shape

    z0 = np.random.randint(0, max(D - pd + 1, 1))
    y0 = np.random.randint(0, max(H - ph + 1, 1))
    x0 = np.random.randint(0, max(W - pw + 1, 1))

    z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw

    if has_channel:
        patch = arr[:, z0:z1, y0:y1, x0:x1]
    else:
        patch = arr[z0:z1, y0:y1, x0:x1]

    return patch


def sample_patch_gaussian(thumbnail, xs, ys, zs, chunk_size, rng, half, center_sigma_ratio=0.25):
    cx, cy, cz = xs // 2, ys // 2, zs // 2

    sx = max(1.0, xs * center_sigma_ratio)
    sy = max(1.0, ys * center_sigma_ratio)
    sz = max(1.0, zs * center_sigma_ratio)
    
    xc = int(np.round(rng.normal(cx, sx)))
    yc = int(np.round(rng.normal(cy, sy)))
    zc = int(np.round(rng.normal(cz, sz)))

    if not (half <= xc < xs-half and half <= yc < ys-half and half <= zc < zs-half):
        return None

    tv = float(thumbnail[xc // chunk_size, yc // chunk_size, zc // chunk_size])

    return xc, yc, zc, tv


def sample_patch_gaussian_uniform(thumbnail, xs, ys, zs, chunk_size, half, rng, center_sigma_ratio=0.25):
    cx, cy, cz = xs // 2, ys // 2, zs // 2
    
    sy = max(1.0, ys * center_sigma_ratio)
    sz = max(1.0, zs * center_sigma_ratio)
    
    xc = int(rng.integers(half, xs - half))
    yc = int(np.round(rng.normal(cy, sy)))
    zc = int(np.round(rng.normal(cz, sz)))

    if not (half <= xc < xs-half and half <= yc < ys-half and half <= zc < zs-half):
        return None

    tv = float(thumbnail[xc // chunk_size, yc // chunk_size, zc // chunk_size])

    return xc, yc, zc, tv


def norm_patch(p):
    p = p.astype(np.float32)
    m = p.mean()
    s = p.std()
    if s > 0:
        p = (p - m) / s
    else:
        p = p - m
    return p


def compute_padding(size, fixer):
    remain = size % fixer
    if remain == 0:
        return (0, 0)
    pad_total = fixer - remain
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return (pad_before, pad_after)


def extend(array, min_val=0, fixer=32):
    x_pad = compute_padding(array.shape[0], fixer)
    y_pad = compute_padding(array.shape[1], fixer)
    z_pad = compute_padding(array.shape[2], fixer)

    pad_width = [x_pad, y_pad, z_pad]

    extended = np.pad(array, pad_width=pad_width, mode='constant', constant_values=min_val)

    return extended


def half_patch(array):
    xs, ys, zs = array.shape
    cx, cy, cz = xs // 2, ys // 2, zs // 2

    return array[0:cx, :, :]


def quarter_patch(array):
    xs, ys, zs = array.shape
    cx, cy, cz = xs // 4, ys // 2, zs // 2

    return array[0:cx, :, :]


def custom_patch(array, width=224):
    xs, ys, zs = array.shape
    cx, cy, cz = xs // 2, ys // 2, zs // 2
    half_width = width // 2

    return array[:, cy-half_width:cy+half_width, cz-half_width:cz+half_width]


def _safe_scale(x: np.ndarray, scale):
    if scale is None:
        return x
    return x * np.float32(scale)


def _clip(x: np.ndarray):
    return np.clip(x, 0, 1)


def _clip_nonneg(x: np.ndarray):
    x[x < 0] = 0
    return x


def _to_key(d) -> str:
    f = float(d)
    return str(int(f)) if f == int(f) else str(f)
 
 
def _resolve_files(data_path, file_list, ext):
    """file_list 우선. data_path와 file_list 둘 다 있으면 join, 없으면 디렉토리 전체."""
    if file_list is not None:
        if data_path is not None:
            # 파일명만 저장된 경우 data_path와 합침, 이미 절대경로면 그대로 사용
            return natsorted([
                f if os.path.isabs(f) else os.path.join(data_path, f)
                for f in file_list
            ])
        return natsorted(file_list)
    elif data_path is not None:
        return natsorted([
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(ext)
        ])
    else:
        raise ValueError("data_path 또는 file_list 중 하나는 필요합니다.")


def collate_dose_dict(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """dict list → dict of batched tensors"""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


class MultiDoseTrain(torch.utils.data.Dataset):
    """
    기존 Train 클래스 구조 그대로 유지하면서 dose_levels를 동적으로 처리.
 
    성능 개선:
      - h5 파일 캐싱 (매 __getitem__마다 열고 닫지 않음)
      - thumbnail 캐싱
      - HDF5 직접 슬라이싱 (전체 볼륨 로드 → 필요한 patch만 읽음)
 
    h5 구조:
      f["thumbnail"]["img"]       ← patch sampling용
      f[drf_key]["img"]           ← 실제 이미지 (drf_key = "1", "4", "10", ...)
      f[drf_key].attrs["scale"]   ← 스케일 팩터
      f[drf_key].attrs["p9999"]   ← 정규화 기준 (ld 기준으로 통일)
    """
 
    def __init__(
        self,
        data_path: str = None,
        patch_size: int = 64,
        patches_per_vol: int = 2,
        seed: int = 0,
        ext: str = "h5",
        dose_levels: List[str] = None,
        nd: str = "1",
        ref: str = None,
        normalize: str = "p9999",
        max_trials: int = 10,
        thumbnail_eps: float = 1e-8,
        normalize_eps: float = NORM_EPS,
        file_list: List[str] = None,
        cache_files: int = 8,            # 동시에 열어둘 h5 파일 수
        thumb_cache_limit: int = 512,    # thumbnail 캐시 한도
    ):
        self.data_path       = data_path
        self.patch_size      = patch_size
        self.patches_per_vol = patches_per_vol
        self.seed            = seed
        self.normalize       = normalize
        self.max_trials      = max_trials
        self.thumbnail_eps   = thumbnail_eps
        self.normalize_eps   = normalize_eps
 
        self.dose_levels    = [_to_key(d) for d in (dose_levels or ["50"])]
        self.nd             = _to_key(nd)
        self.ref            = _to_key(ref) if ref is not None else self.dose_levels[0]

        self.files          = _resolve_files(data_path, file_list, ext)
 
        # ── 파일 캐시 (LRU) ──────────────────────────────────
        self.cache_files    = max(1, int(cache_files))

        self._file_cache: Dict[str, h5py.File] = {}
        self._file_lru:   List[str]            = []
 
        # ── thumbnail 캐시 ────────────────────────────────────
        self.thumb_cache_limit = int(thumb_cache_limit)
        self._thumb_cache: Dict[str, tuple]    = {}

        # ── epoch별 RNG 다양성을 위한 call counter ────────────
        self._call_count = 0
 
    def __len__(self):
        return len(self.files) * self.patches_per_vol

    def close(self):
        for f in list(self._file_cache.values()):
            try:
                f.close()
            except Exception:
                pass
        self._file_cache.clear()
        self._file_lru.clear()

    def __del__(self):
        self.close()
 
    # ── 파일 캐시 헬퍼 ───────────────────────────────────────
 
    def _evict_one_file(self):
        if not self._file_lru:
            return
        old_path = self._file_lru.pop(0)
        f = self._file_cache.pop(old_path, None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
 
    def _get_file(self, h5_path: str) -> h5py.File:
        f = self._file_cache.get(h5_path)
        if f is not None:
            try:
                self._file_lru.remove(h5_path)
            except ValueError:
                pass
            self._file_lru.append(h5_path)
            return f

        if len(self._file_cache) >= self.cache_files:
            self._evict_one_file()

        f = h5py.File(h5_path, "r")
        self._file_cache[h5_path] = f
        self._file_lru.append(h5_path)
        return f
 
    def _get_thumbnail(self, f: h5py.File, h5_path: str):
        cached = self._thumb_cache.get(h5_path)
        if cached is not None:
            return cached
 
        thumb      = f["thumbnail"]["img"][...].astype(np.float32)
        chunk_size = int(f["thumbnail"].attrs["chunk_size"])
 
        if len(self._thumb_cache) >= self.thumb_cache_limit:
            self._thumb_cache.pop(next(iter(self._thumb_cache)))
        self._thumb_cache[h5_path] = (thumb, chunk_size)
        return thumb, chunk_size
 
    # ── 핵심 ─────────────────────────────────────────────────
 
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx  = idx // self.patches_per_vol
        patch_idx = idx %  self.patches_per_vol
 
        h5_path = self.files[file_idx]
        half    = self.patch_size // 2
        ps      = self.patch_size

        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info is not None else 0
        self._call_count += 1
        rng = np.random.default_rng(
            self.seed
            + worker_id   * 10_000_000_000
            + self._call_count * 1_000_000
            + file_idx    * 100_000
            + patch_idx
        )
 
        f = self._get_file(h5_path)
 
        thumbnail, chunk_size = self._get_thumbnail(f, h5_path)
        xs, ys, zs = f[self.nd]["img"].shape
 
        # ── patch center 샘플링 ──────────────────────────────
        xc = yc = zc = None
        for _ in range(self.max_trials):
            sampled = sample_patch_gaussian_uniform(
                thumbnail, xs, ys, zs, chunk_size, half, rng
            )
            if sampled is not None and sampled[3] > self.thumbnail_eps:
                xc, yc, zc, _ = sampled
                break
 
        if xc is None:
            xc = int(rng.integers(half, xs - half))
            yc = int(rng.integers(half, ys - half))
            zc = int(rng.integers(half, zs - half))
 
        x0, x1 = xc - half, xc + half
        y0, y1 = yc - half, yc + half
        z0, z1 = zc - half, zc + half
 
        # 경계 보정
        if x0 < 0 or y0 < 0 or z0 < 0 or x1 > xs or y1 > ys or z1 > zs:
            xc, yc, zc = xs // 2, ys // 2, zs // 2
            x0, x1 = xc - half, xc + half
            y0, y1 = yc - half, yc + half
            z0, z1 = zc - half, zc + half
 
        # ── 정규화 기준 ──────────────────────────────────────
        vmax    = float(f[self.ref].attrs[self.normalize])
        denom   = vmax + self.normalize_eps
 
        # ── 각 dose level patch 로드 (직접 슬라이싱) ─────────
        def load_patch(drf_key: str) -> torch.Tensor:
            grp   = f[str(drf_key)]
            scale = float(grp.attrs.get("scale", 1.0))
 
            # ★ 핵심: [...]로 전체 로드하지 않고 슬라이싱으로 필요한 부분만 읽음
            patch = grp["img"][x0:x1, y0:y1, z0:z1].astype(np.float32)
 
            if patch.shape != (ps, ps, ps):
                # 경계 케이스 fallback
                xc2, yc2, zc2 = xs // 2, ys // 2, zs // 2
                patch = grp["img"][
                    xc2 - half : xc2 + half,
                    yc2 - half : yc2 + half,
                    zc2 - half : zc2 + half,
                ].astype(np.float32)
 
            patch = np.clip(patch * scale / denom, 0.0, 1.0)
            return torch.from_numpy(patch).unsqueeze(0).float()   # [1,D,H,W]
 
        out = {}
        for drf in set(self.dose_levels) | {self.nd}:
            out[str(drf)] = load_patch(str(drf))

        return out


class MultiDoseTest(Dataset):
    def __init__(self, data_path=None, file_list=None, ext="h5",
                 axial_patch_size=224, fixer_size=8, ld="50", nd="1", ref="50",
                 normalize="p9999", normalize_eps=NORM_EPS):

        self.files = _resolve_files(data_path, file_list, ext)

        self.data_path        = data_path
        self.axial_patch_size = axial_patch_size
        self.fixer_size       = fixer_size
        self.normalize        = normalize
        self.normalize_eps    = normalize_eps

        self.ld  = _to_key(ld)
        self.nd  = _to_key(nd)
        self.ref = _to_key(ref)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        name = os.path.splitext(os.path.basename(h5_path))[0]

        with h5py.File(h5_path, "r") as f:
            grp_ld  = f[self.ld]
            grp_nd  = f[self.nd]
            grp_ref = f[self.ref]

            ld_array = grp_ld["img"][...].astype(np.float32)
            nd_array = grp_nd["img"][...].astype(np.float32)

            ld_array = _safe_scale(ld_array, grp_ld.attrs.get("scale"))
            nd_array = _safe_scale(nd_array, grp_nd.attrs.get("scale"))

            vmax_ref = float(grp_ref.attrs[self.normalize])

        ld_array = custom_patch(ld_array, width=self.axial_patch_size)  # (xs, 224, 224)
        nd_array = custom_patch(nd_array, width=self.axial_patch_size)

        # ld_array = extend(ld_array, min_val=0, fixer=self.fixer_size)   # axial dim 패딩
        # nd_array = extend(nd_array, min_val=0, fixer=self.fixer_size)

        denom = vmax_ref + self.normalize_eps

        ld_array = ld_array / denom
        nd_array = nd_array / denom

        ld_array = _clip(ld_array)
        nd_array = _clip(nd_array)

        ld_tensor = torch.from_numpy(ld_array).unsqueeze(0).to(torch.float32)
        nd_tensor = torch.from_numpy(nd_array).unsqueeze(0).to(torch.float32)

        return ld_tensor, nd_tensor, vmax_ref, name