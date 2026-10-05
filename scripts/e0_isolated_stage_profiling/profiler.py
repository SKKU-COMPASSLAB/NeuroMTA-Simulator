import math
import re
from contextlib import contextmanager

import cv2
import numpy as np
import torch


SYSTEM_PROMPT = (
    "You are watching a live video stream. Questions arrive in real time; "
    "answer each using everything you have seen so far."
)


@contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


class NvtxModuleRange:
    def __init__(self, module, range_name: str):
        self.module = module
        self.range_name = range_name
        self.handles = []

    def __enter__(self):
        def pre_hook(_module, _inputs):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(self.range_name)

        def post_hook(_module, _inputs, _output):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()

        self.handles.append(self.module.register_forward_pre_hook(pre_hook))
        self.handles.append(self.module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class DecoderStackNvtxRange:
    def __init__(self, first_layer, last_layer, range_name: str):
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.range_name = range_name
        self.handles = []

    def __enter__(self):
        def pre_hook(_module, _inputs):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(self.range_name)

        def post_hook(_module, _inputs, _output):
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()

        self.handles.append(self.first_layer.register_forward_pre_hook(pre_hook))
        self.handles.append(self.last_layer.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def pick_first_module(model, name_patterns, class_patterns=()):
    matches = []
    compiled_names = [re.compile(p) for p in name_patterns]
    compiled_classes = [re.compile(p) for p in class_patterns]
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if any(p.search(name) for p in compiled_names) or any(p.search(class_name) for p in compiled_classes):
            matches.append((name, module))
    if not matches:
        return None, None
    matches.sort(key=lambda item: (item[0].count("."), len(item[0])))
    return matches[0]


def pick_decoder_layer_stack(model):
    layer_matches = []
    layer_name_re = re.compile(r"(^|\.)(layers|h|blocks)\.\d+$")
    layer_class_re = re.compile(r"(DecoderLayer|Block|TransformerLayer)$")
    non_llm_re = re.compile(r"(vision|visual|video|image|audio|talker|codec)", re.IGNORECASE)
    for name, module in model.named_modules():
        if non_llm_re.search(name):
            continue
        if layer_name_re.search(name) or layer_class_re.search(module.__class__.__name__):
            layer_matches.append((name, module))
    if not layer_matches:
        return None, None, None, None

    def layer_index(item):
        match = re.search(r"\.(\d+)$", item[0])
        return int(match.group(1)) if match else 0

    layer_matches.sort(key=layer_index)
    first_name, first_module = layer_matches[0]
    last_name, last_module = layer_matches[-1]
    return first_name, first_module, last_name, last_module


@contextmanager
def epd_stage_nvtx_ranges(model):
    vision_name, vision_module = pick_first_module(
        model,
        name_patterns=[
            r"(^|\.)visual($|\.)",
            r"(^|\.)vision($|\.)",
            r"vision_tower",
            r"visual_encoder",
            r"video_tower",
        ],
        class_patterns=[r"Vision", r"Visual"],
    )
    first_name, first_layer, last_name, last_layer = pick_decoder_layer_stack(model)
    contexts = []
    if vision_module is not None:
        contexts.append(NvtxModuleRange(vision_module, "VISION_ENCODING"))
    if first_layer is not None and last_layer is not None:
        contexts.append(DecoderStackNvtxRange(first_layer, last_layer, "LLM_PREFILL_VISION_TEXT"))

    print(f"[nvtx] VISION_ENCODING module: {vision_name or 'not found'}")
    print(f"[nvtx] LLM_PREFILL_VISION_TEXT layer stack: {first_name or 'not found'} -> {last_name or 'not found'}")
    try:
        for ctx in contexts:
            ctx.__enter__()
        yield
    finally:
        for ctx in reversed(contexts):
            ctx.__exit__(None, None, None)


def decode_interval_chunks(video_path, start_sec, end_sec, target_fps, chunk_seconds):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0:
        src_fps = 30.0

    frame_interval = max(int(round(src_fps / target_fps)), 1)
    chunks = []
    cur_start = start_sec
    cur_frames = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        tsec = frame_idx / src_fps
        if tsec > end_sec:
            break
        if tsec > start_sec and frame_idx % frame_interval == 0:
            cur_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        while tsec >= cur_start + chunk_seconds:
            if cur_frames:
                chunks.append((cur_start, min(cur_start + chunk_seconds, end_sec), cur_frames))
                cur_frames = []
            cur_start += chunk_seconds
        frame_idx += 1
    cap.release()

    if cur_frames:
        chunks.append((cur_start, end_sec, cur_frames))
    if not chunks:
        chunks.append((start_sec, end_sec, [np.zeros((360, 640, 3), dtype=np.uint8)]))

    fixed = []
    for chunk_start, chunk_end, frames in chunks:
        if len(frames) % 2 == 1:
            frames = frames + [frames[-1]]
        while len(frames) < 2:
            frames.append(frames[-1])
        fixed.append((chunk_start, chunk_end, np.stack(frames, axis=0).astype(np.uint8)))
    return fixed


def move_inputs_to_device(inputs, device):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def video_patch_tokens(inputs):
    grid = inputs.get("video_grid_thw")
    if grid is None:
        return 0
    return int(torch.prod(grid, dim=1).sum().item())


def merged_visual_tokens(inputs, spatial_merge_size=2):
    merge = spatial_merge_size * spatial_merge_size
    return int(math.ceil(video_patch_tokens(inputs) / merge))


def build_inputs(processor, video_array, text, device, include_system=False, add_generation_prompt=False):
    conversation = []
    if include_system:
        conversation.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
    conversation.append({
        "role": "user",
        "content": [
            {"type": "video", "video": video_array},
            {"type": "text", "text": text},
        ],
    })
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
        load_audio_from_video=False,
        processor_kwargs={"text_kwargs": {"padding": True}},
    )
    return move_inputs_to_device(inputs, device)


def metadata_range_name(prefix, fields):
    pieces = [prefix]
    for key, value in fields.items():
        text = str(value).replace("|", "_").replace(" ", "_")
        pieces.append(f"{key}={text}")
    return "|".join(pieces)
