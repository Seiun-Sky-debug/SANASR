import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OSEDiffDegradation:
    """生成训练用的真实退化低清图像。"""

    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor

    @staticmethod
    def _sanitize(img):
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        return np.clip(img, 0.0, 1.0)

    @staticmethod
    def _random_blur(img):
        if random.random() > 0.8:
            return OSEDiffDegradation._sanitize(img)
        k = random.choice([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
        sigma = random.uniform(0.2, 3.0)
        out = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
        return OSEDiffDegradation._sanitize(out)

    @staticmethod
    def _random_resize(img):
        h, w = img.shape[:2]
        mode = random.choices(["up", "down", "keep"], weights=[0.2, 0.7, 0.1], k=1)[0]
        if mode == "up":
            scale = random.uniform(1.0, 1.5)
        elif mode == "down":
            scale = random.uniform(0.15, 1.0)
        else:
            scale = 1.0
        oh = max(8, int(h * scale))
        ow = max(8, int(w * scale))
        interp = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        out = cv2.resize(img, (ow, oh), interpolation=interp)
        return OSEDiffDegradation._sanitize(out)

    @staticmethod
    def _random_noise(img):
        img = OSEDiffDegradation._sanitize(img)
        if random.random() < 0.5:
            sigma = random.uniform(1.0, 30.0) / 255.0
            noise = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
            out = np.clip(img + noise, 0.0, 1.0)
        else:
            img = OSEDiffDegradation._sanitize(img)
            vals = 2 ** random.uniform(2, 8)
            lam = np.clip(img * vals, 0.0, 1e4).astype(np.float64)
            out = (np.random.poisson(lam) / vals).astype(np.float32)
            out = np.clip(out, 0.0, 1.0)
        return OSEDiffDegradation._sanitize(out)

    @staticmethod
    def _random_jpeg(img):
        img = OSEDiffDegradation._sanitize(img)
        q = random.randint(30, 95)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        _, enc = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return OSEDiffDegradation._sanitize(dec)

    def __call__(self, hq_tensor):
        _, h, w = hq_tensor.shape
        img = self._sanitize(hq_tensor.permute(1, 2, 0).numpy().astype(np.float32))
        img = self._random_blur(img)
        img = self._random_resize(img)
        img = self._random_noise(img)
        img = self._random_jpeg(img)
        img = self._random_blur(img)
        img = self._random_resize(img)
        img = self._random_noise(img)
        img = self._random_jpeg(img)
        lh = max(8, h // self.scale_factor)
        lw = max(8, w // self.scale_factor)
        img = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = self._sanitize(img)
        return torch.from_numpy(img).permute(2, 0, 1).float()


class SRDataset(Dataset):
    """读取配对数据或在线生成低清训练样本。"""

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        hq_dir,
        lq_dir=None,
        image_size=512,
        scale_factor=4,
        is_train=True,
        use_osediff_degradation=False,
    ):
        super().__init__()
        self.hq_dir = Path(hq_dir)
        self.lq_dir = None
        if lq_dir:
            self.lq_dir = Path(lq_dir)
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.use_degradation = self.lq_dir is None or not self.lq_dir.exists() or lq_dir == ""
        self.use_osediff_degradation = use_osediff_degradation
        self.osediff_degrader = OSEDiffDegradation(scale_factor=scale_factor)
        self.max_resample_attempts = 20
        self._bad_files_reported = set()

        self.hq_files = sorted([f for f in self.hq_dir.iterdir() if f.suffix.lower() in self.IMAGE_EXTS])

        if not self.use_degradation:
            lq_all = sorted([f for f in self.lq_dir.iterdir() if f.suffix.lower() in self.IMAGE_EXTS])
            self.lq_files = self._match_pairs(self.hq_files, lq_all)
            assert len(self.hq_files) == len(self.lq_files), (
                f"Mismatch: {len(self.hq_files)} HQ vs {len(self.lq_files)} LQ images. "
                f"Check filenames in {hq_dir} and {lq_dir}."
            )

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _match_pairs(hq_files, lq_files):
        lq_map = {f.stem: f for f in lq_files}
        matched = []
        for hf in hq_files:
            if hf.stem in lq_map:
                matched.append(lq_map[hf.stem])
            else:
                alt_stem = hf.stem.replace("_HR", "").replace("_hr", "")
                for lq_stem, lq_f in lq_map.items():
                    clean = lq_stem.replace("_LR4", "").replace("_LR2", "").replace("_lr", "")
                    if alt_stem == clean:
                        matched.append(lq_f)
                        break

        if len(matched) == len(hq_files):
            return matched
        if len(hq_files) == len(lq_files):
            return lq_files
        raise ValueError(
            f"Cannot match HQ ({len(hq_files)}) and LQ ({len(lq_files)}) images by name or count."
        )

    def __len__(self):
        return len(self.hq_files)

    def _load_image(self, path):
        with Image.open(path) as img:
            return img.convert("RGB")

    def _random_crop_pair(self, hq_img, lq_img):
        hq_w, hq_h = hq_img.size
        crop_size = self.image_size

        if hq_w < crop_size or hq_h < crop_size:
            hq_img = hq_img.resize((max(hq_w, crop_size), max(hq_h, crop_size)), Image.LANCZOS)
            hq_w, hq_h = hq_img.size
            if lq_img is not None:
                lq_w = hq_w // self.scale_factor
                lq_h = hq_h // self.scale_factor
                lq_img = lq_img.resize((lq_w, lq_h), Image.LANCZOS)

        top = random.randint(0, hq_h - crop_size)
        left = random.randint(0, hq_w - crop_size)
        hq_crop = TF.crop(hq_img, top, left, crop_size, crop_size)

        if lq_img is not None:
            lq_w, lq_h = lq_img.size
            actual_scale_x = hq_w / lq_w
            actual_scale_y = hq_h / lq_h

            if abs(actual_scale_x - 1.0) < 0.01:
                lq_crop = TF.crop(lq_img, top, left, crop_size, crop_size)
            else:
                lq_crop_size = crop_size // round(actual_scale_x)
                lq_top = round(top / actual_scale_y)
                lq_left = round(left / actual_scale_x)
                lq_top = min(lq_top, lq_h - lq_crop_size)
                lq_left = min(lq_left, lq_w - lq_crop_size)
                lq_crop = TF.crop(lq_img, lq_top, lq_left, lq_crop_size, lq_crop_size)
                lq_crop = lq_crop.resize((crop_size, crop_size), Image.BICUBIC)
        else:
            lq_crop = None

        return hq_crop, lq_crop

    def _degrade(self, hq_tensor):
        if self.use_osediff_degradation:
            return self.osediff_degrader(hq_tensor)

        _, H, W = hq_tensor.shape
        lq_h, lq_w = H // self.scale_factor, W // self.scale_factor
        img_np = (hq_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_small = cv2.resize(img_np, (lq_w, lq_h), interpolation=cv2.INTER_CUBIC)
        noise_sigma = random.uniform(0, 25)
        noise = np.random.normal(0, noise_sigma, img_small.shape).astype(np.float32)
        img_small = np.clip(img_small.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        quality = random.randint(30, 95)
        _, enc = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img_small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        img_up = cv2.resize(img_small, (W, H), interpolation=cv2.INTER_CUBIC)
        img_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img_up.astype(np.float32) / 255.0).permute(2, 0, 1)

    def __getitem__(self, idx):
        cur_idx = idx
        for _ in range(self.max_resample_attempts):
            try:
                hq_img = self._load_image(self.hq_files[cur_idx])

                use_online_deg = self.use_degradation or self.use_osediff_degradation
                if use_online_deg:
                    if self.is_train:
                        hq_img, _ = self._random_crop_pair(hq_img, None)
                        if random.random() < 0.5:
                            hq_img = TF.hflip(hq_img)
                    else:
                        hq_img = hq_img.resize((self.image_size, self.image_size), Image.LANCZOS)

                    hq_tensor = self.to_tensor(hq_img)
                    lq_tensor = self._degrade(hq_tensor)
                else:
                    lq_img = self._load_image(self.lq_files[cur_idx])

                    if self.is_train:
                        hq_img, lq_img = self._random_crop_pair(hq_img, lq_img)
                        if random.random() < 0.5:
                            hq_img = TF.hflip(hq_img)
                            lq_img = TF.hflip(lq_img)
                    else:
                        hq_img = hq_img.resize((self.image_size, self.image_size), Image.LANCZOS)
                        lq_img = lq_img.resize((self.image_size, self.image_size), Image.BICUBIC)

                    hq_tensor = self.to_tensor(hq_img)
                    lq_tensor = self.to_tensor(lq_img)

                hq_tensor = hq_tensor * 2.0 - 1.0
                lq_tensor = lq_tensor * 2.0 - 1.0
                return {"hq": hq_tensor, "lq": lq_tensor, "filename": self.hq_files[cur_idx].stem}
            except (OSError, UnidentifiedImageError, ValueError) as e:
                bad_path = str(self.hq_files[cur_idx])
                if bad_path not in self._bad_files_reported:
                    self._bad_files_reported.add(bad_path)
                    print(f"[WARN] Skip unreadable image: {bad_path} ({type(e).__name__})")
                cur_idx = random.randint(0, len(self.hq_files) - 1)

        raise RuntimeError(
            f"Failed to fetch a valid sample after {self.max_resample_attempts} attempts. "
            "Please check dataset integrity."
        )
