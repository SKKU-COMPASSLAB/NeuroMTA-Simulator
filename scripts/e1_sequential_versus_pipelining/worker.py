import argparse
import sys
from contextlib import contextmanager

import torch
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from scripts.e1_sequential_versus_pipelining.dataset import load_questions, render_prompt, streamingbench_video_path
from scripts.e1_sequential_versus_pipelining.profiler import (
    build_inputs,
    decode_interval_chunks,
    DecoderStackNvtxRange,
    NvtxModuleRange,
    merged_visual_tokens,
    metadata_range_name,
    nvtx_range,
    pick_decoder_layer_stack,
    pick_vision_module,
    video_patch_tokens,
)


@contextmanager
def module_or_plain_nvtx(module_ctx_factory, module_args, name):
    if any(arg is None for arg in module_args):
        with nvtx_range(name):
            yield
        return
    with module_ctx_factory(*module_args, name):
        yield


def decode_tokens(model, decoder_modules, prefill_out, max_new_tokens, sample_fields, context_tokens_before_decode):
    past_key_values = prefill_out.past_key_values
    cur_token = torch.argmax(prefill_out.logits[:, -1:, :], dim=-1)
    with nvtx_range(metadata_range_name("DECODE_STAGE", sample_fields)):
        for step in range(max_new_tokens):
            step_fields = dict(sample_fields)
            step_fields.update(
                {
                    "decode_step": step,
                    "context_tokens": context_tokens_before_decode + step,
                    "generated_tokens": 1,
                }
            )
            with module_or_plain_nvtx(DecoderStackNvtxRange, decoder_modules, metadata_range_name("DECODE_STEP", step_fields)):
                out = model(
                    input_ids=cur_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = out.past_key_values
            cur_token = torch.argmax(out.logits[:, -1:, :], dim=-1)
    return past_key_values


def profile_video(args):
    assert torch.cuda.is_available(), "CUDA GPU is required for profiling."
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")

    questions = load_questions(args.data_dir, args.sample_id)
    video_path = streamingbench_video_path(args.data_dir, args.sample_id)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    vision_name, vision_module = pick_vision_module(model)
    first_name, first_layer, last_name, last_layer = pick_decoder_layer_stack(model)
    print(f"[nvtx] ENCODE_STEP module: {vision_name or 'not found'}")
    print(f"[nvtx] PREFILL/DECODE stack: {first_name or 'not found'} -> {last_name or 'not found'}")
    vision_modules = (vision_module,)
    decoder_modules = (first_layer, last_layer)

    past_key_values = None
    prev_sec = 0.0
    total_frames_seen = 0
    context_tokens = 0

    print(f"[worker] sample_id={args.sample_id} questions={len(questions)} video={video_path}")
    for question in questions:
        chunks = decode_interval_chunks(
            video_path=video_path,
            start_sec=prev_sec,
            end_sec=question.timestamp,
            target_fps=args.fps,
            chunk_seconds=args.chunk_seconds,
        )
        prev_sec = question.timestamp

        sample_patch_tokens = 0
        sample_merged_tokens = 0
        sample_prefill_tokens = 0
        sample_frames = sum(len(chunk[2]) for chunk in chunks)
        total_frames_seen += sample_frames
        prefill_out = None

        sample_fields = {
            "sample_id": question.sample_id,
            "question_id": question.question_id,
            "sample_index": question.sample_index,
            "task_type": question.task_type.replace(" ", "_"),
            "timestamp_sec": int(question.timestamp),
            "frame_index": total_frames_seen,
            "sample_video_frames": sample_frames,
            "decode_tokens": args.max_new_tokens,
        }

        with nvtx_range(metadata_range_name("SAMPLE", sample_fields)):
            for chunk_index, (chunk_start, chunk_end, video_array) in enumerate(chunks):
                is_first_chunk = question.sample_index == 0 and chunk_index == 0
                is_last_chunk = chunk_index == len(chunks) - 1
                if is_last_chunk:
                    text = render_prompt(question)
                else:
                    text = (
                        f"[stream {chunk_start:.1f}s-{chunk_end:.1f}s] "
                        "Continue watching this live video segment. Do not answer yet."
                    )
                inputs = build_inputs(
                    processor=processor,
                    video_array=video_array,
                    text=text,
                    device=device,
                    include_system=is_first_chunk,
                    add_generation_prompt=is_last_chunk,
                )
                chunk_patch_tokens = video_patch_tokens(inputs)
                chunk_merged_tokens = merged_visual_tokens(inputs, model.spatial_merge_size)
                chunk_prefill_tokens = int(inputs["input_ids"].shape[1])
                sample_patch_tokens += chunk_patch_tokens
                sample_merged_tokens += chunk_merged_tokens
                sample_prefill_tokens += chunk_prefill_tokens
                context_tokens += chunk_prefill_tokens

                chunk_fields = dict(sample_fields)
                chunk_fields.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_start": f"{chunk_start:.1f}",
                        "chunk_end": f"{chunk_end:.1f}",
                        "frames": len(video_array),
                        "patch_tokens": chunk_patch_tokens,
                        "merged_visual_tokens": chunk_merged_tokens,
                        "prefill_tokens": chunk_prefill_tokens,
                        "context_tokens_after_prefill": context_tokens,
                    }
                )

                forward_inputs = dict(inputs)
                if past_key_values is not None:
                    forward_inputs["past_key_values"] = past_key_values

                with nvtx_range(metadata_range_name("STREAM_CHUNK", chunk_fields)):
                    with module_or_plain_nvtx(NvtxModuleRange, vision_modules, metadata_range_name("ENCODE_STEP", chunk_fields)):
                        with module_or_plain_nvtx(DecoderStackNvtxRange, decoder_modules, metadata_range_name("PREFILL_STAGE", chunk_fields)):
                            prefill_out = model(
                                **forward_inputs,
                                use_cache=True,
                                return_dict=True,
                            )
                past_key_values = prefill_out.past_key_values

            with nvtx_range(metadata_range_name(
                "SAMPLE_META",
                {
                    "sample_id": question.sample_id,
                    "sample_index": question.sample_index,
                    "vision_patch_tokens": sample_patch_tokens,
                    "merged_visual_tokens": sample_merged_tokens,
                    "prefill_tokens": sample_prefill_tokens,
                    "context_tokens": context_tokens,
                },
            )):
                pass

            if prefill_out is None:
                raise RuntimeError("No prefill output was produced.")
            decode_fields = dict(sample_fields)
            decode_fields.update(
                {
                    "vision_patch_tokens": sample_patch_tokens,
                    "merged_visual_tokens": sample_merged_tokens,
                    "prefill_tokens": sample_prefill_tokens,
                    "context_tokens": context_tokens,
                }
            )
            past_key_values = decode_tokens(model, decoder_modules, prefill_out, args.max_new_tokens, decode_fields, context_tokens)
            context_tokens += args.max_new_tokens

        print(
            f"[worker] q{question.sample_index}: t={question.timestamp_text}, "
            f"frames_seen={total_frames_seen}, prefill_tokens={sample_prefill_tokens}, "
            f"context={context_tokens}"
        )
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    profile_video(args)


if __name__ == "__main__":
    main()
