"""CPU regression checks; CUDA numeric checks run when a GPU is available."""
import gc
import unittest
import weakref
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from inference.decoders.tiny import TinyDecoder
from inference.preprocessing.input import pil_center_crop_resize
from inference.preprocessing.resident import (
    padded_frame_capacity, prepare_rgb8, resident_bytes,
)
from inference.preprocessing.temporal import expand_for_temporal_compression
from inference.preprocessing.padding import repeat_last_frame


class ResidentInputTests(unittest.TestCase):
    def test_pil_normalization_tail_and_shared_storage_match_stock(self):
        for count in [1, 5, 8, 17, 144]:
            frames = np.random.default_rng(count).integers(
                0, 256, (count, 13, 17, 3), dtype=np.uint8)
            actual = prepare_rgb8(frames, 16, 12, device="cpu")
            resized = pil_center_crop_resize(frames, 16, 12,
                                            resample=Image.Resampling.BICUBIC)
            expected = torch.from_numpy(resized).permute(3, 0, 1, 2).unsqueeze(0)
            expected = expected.float().div_(127.5).sub_(1).to(torch.bfloat16)
            self.assertTrue(torch.equal(actual.video, expected))
            self.assertTrue(np.array_equal(actual.reference_rgb8, resized))
            expanded = expand_for_temporal_compression(expected, 4)
            padded = repeat_last_frame(expanded.video, actual.padded.shape[2])
            self.assertTrue(torch.equal(actual.padded, padded))
            self.assertEqual(actual.video.untyped_storage().data_ptr(),
                             actual.padded.untyped_storage().data_ptr())
            self.assertEqual(actual.padded.numel() * actual.padded.element_size(),
                             resident_bytes(count, 16, 12))
        self.assertEqual(padded_frame_capacity(144), 161)

    def test_budget_rejects_before_allocating(self):
        frames = np.zeros((144, 2, 2, 3), dtype=np.uint8)
        with patch("torch.empty", side_effect=AssertionError("must not allocate")):
            with self.assertRaisesRegex(ValueError, "exceeding its budget"):
                prepare_rgb8(frames, 2688, 1536, max_resident_bytes=1)

    def test_no_resize_tensor_path_is_exact(self):
        frames = np.random.default_rng(4).integers(0, 256, (5, 16, 12, 3), dtype=np.uint8)
        exact = prepare_rgb8(frames, 16, 12, device="cpu")
        tensor = prepare_rgb8(frames, 16, 12, device="cpu", tensor_resize=True)
        self.assertTrue(torch.equal(exact.video, tensor.video))
        self.assertEqual(tensor.resize_control["maxRgb8Difference"], 0)

    def test_invalid_input_rejected(self):
        for frames in [np.zeros((0, 4, 4, 3), dtype=np.uint8),
                       np.zeros((1, 4, 4, 3), dtype=np.float32),
                       np.zeros((1, 4, 4, 4), dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                prepare_rgb8(frames, 4, 4, device="cpu")
        frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
        for options in [dict(batch_frames=0), dict(height=0), dict(width=-1)]:
            args = dict(height=4, width=4, device="cpu", **{})
            args.update(options)
            with self.assertRaises(ValueError):
                prepare_rgb8(frames, **args)


class DecoderTests(unittest.TestCase):
    def test_exact_values_counts_and_history_for_partial_chunks(self):
        class Module:
            def __init__(self):
                self.calls = []

            def decode_video(self, latents, *, parallel, show_progress_bar, cond):
                self.calls.append((len(latents[0]), cond.clone(), parallel))
                return (cond * .5).transpose(1, 2)

        for dtype in [torch.float32, torch.bfloat16]:
            for first in [True, False]:
                for parallel in [True, False]:
                    for count in [1, 4, 5, 9]:
                        module = Module()
                        decoder = TinyDecoder(module)
                        frame_count = 4 * count - (3 if first else 0)
                        reference = torch.randn(1, 3, frame_count, 8, 8, dtype=dtype)
                        actual = decoder._decode_chunks(
                            torch.randn(1, 2, count, 2, 2, dtype=dtype), reference,
                            device=torch.device("cpu"), dtype=dtype,
                            use_parallel=parallel, first=first,
                        )
                        expected = (reference * .5).mul(2).sub(1)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertEqual(len(module.calls), (count + 3) // 4)
                        self.assertTrue(torch.equal(torch.cat([c[1] for c in module.calls], 2), reference))
                        self.assertTrue(all(c[2] == parallel for c in module.calls))

    def test_oom_retry_releases_failed_chunk_and_normalizes_once(self):
        class Module:
            def decode_video(self, *args, **kwargs):
                raise AssertionError("test replaces chunk execution")

            def parameters(self):
                return iter([torch.zeros(1)])

        decoder = TinyDecoder(Module())
        decoder._auto_candidates = lambda *args: [("parallel", True), ("serial", False)]
        refs = []

        def chunks(*args, use_parallel, **kwargs):
            if use_parallel:
                temporary = torch.ones(1024)
                refs.append(weakref.ref(temporary))
                raise torch.cuda.OutOfMemoryError("controlled test OOM")
            gc.collect()
            self.assertIsNone(refs[0]())
            # _decode_chunks now owns the range transform before transfer.
            return torch.full((1, 3, 1, 8, 8), .25)

        decoder._decode_chunks = chunks
        result = decoder.decode(torch.ones(1, 2, 1, 2, 2),
                                lq_video=torch.ones(1, 3, 1, 8, 8))
        self.assertTrue(torch.equal(result.video, torch.full_like(result.video, .25)))
        self.assertEqual(decoder.last_auto_fallbacks, 1)
        self.assertEqual(decoder.last_execution_mode, "serial")


class OutputTests(unittest.TestCase):
    def test_final_encoder_resize_preserves_frame_count(self):
        from inference.output import IncrementalVideoWriter
        pixels = np.stack([np.full((16, 16, 3), level, dtype=np.uint8)
                           for level in [0, 64, 128, 192, 255]])
        with tempfile.TemporaryDirectory() as folder:
            writer = IncrementalVideoWriter(Path(folder) / "test.mp4", 24,
                                            output_size=(12, 8))
            try:
                writer.append_rgb8(pixels)
            finally:
                writer.close()
            metadata = writer.verify()
        self.assertEqual((metadata["frames"], metadata["width"], metadata["height"]), (5, 12, 8))

    def test_rgb8_frames_pass_to_writer_without_float_roundtrip(self):
        from inference.output import IncrementalVideoWriter
        writer = object.__new__(IncrementalVideoWriter)
        writer.first_frame = None
        writer.frame_count = 0
        received = []
        writer.writer = type("Sink", (), {"append_data": lambda _, frame: received.append(frame.copy())})()
        pixels = np.random.default_rng(5).integers(0, 256, (5, 7, 9, 3), dtype=np.uint8)
        writer.append_rgb8(pixels)
        self.assertTrue(np.array_equal(np.stack(received), pixels))
        self.assertEqual(writer.frame_count, 5)
        self.assertTrue(np.array_equal(writer.first_frame, pixels[0]))
        with self.assertRaises(ValueError):
            writer.append_rgb8(pixels.astype(np.float32))


class FastPipelineTests(unittest.TestCase):
    def test_144_source_frames_survive_padding_and_rgb8_handoff(self):
        from inference.fast import infer_fast
        from inference.request import InferenceRequest

        class Runner:
            def run(self, video, context, *, padded_lq, progress):
                self.input_frames = video.shape[2]
                self.capacity = padded_lq.shape[2]
                self.shared = video.untyped_storage().data_ptr() == padded_lq.untyped_storage().data_ptr()
                # Identity stand-in for the GPU model; no synthetic tail escapes.
                return SimpleNamespace(video=video.cpu(), metadata={})

        runner = Runner()
        config = SimpleNamespace(
            temporal_compression=4, decoder=SimpleNamespace(kind="tiny", decode_every="auto"),
            color_correction=SimpleNamespace(enabled=False, levels=5),
            dtype="bf16", device="cuda:0", seed=1,
        )
        pipeline = SimpleNamespace(config=config, runner=runner)
        frames = np.zeros((144, 128, 128, 3), dtype=np.uint8)
        frames[143] = 255
        request = InferenceRequest(input_path="unused-for-raw-input.mp4", fps=24,
                                   height=128, width=128)
        # Exercise the actual orchestration on CPU; only CUDA device selection
        # and synchronization are replaced, not preparation/colour/frame logic.
        with patch("inference.fast._fast_device", return_value=torch.device("cpu")), \
                patch("torch.cuda.synchronize"):
            result = infer_fast(pipeline, request, rgb8=frames)
        self.assertEqual((runner.input_frames, runner.capacity), (145, 161))
        self.assertTrue(runner.shared)
        self.assertEqual(result.frames, 144)
        self.assertEqual(result.fps, 24)
        self.assertIsNone(result.output_path)
        self.assertTrue(np.array_equal(result.frames_rgb8, frames))

    def test_unsupported_temporal_settings_fail_before_preprocessing(self):
        from inference.fast import infer_fast
        from inference.request import InferenceRequest
        pipeline = SimpleNamespace(config=SimpleNamespace(
            temporal_compression=2, decoder=SimpleNamespace(kind="tiny", decode_every="auto")))
        with self.assertRaisesRegex(ValueError, "TC4"):
            infer_fast(pipeline, InferenceRequest(input_path="unused.mp4"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA numeric checks require GPU")
class CudaTests(unittest.TestCase):
    def test_wavelet_matches_stock_fp32(self):
        from inference.postprocessing.wavelet_cuda import validate_wavelet
        validate_wavelet(torch.device("cuda:0"))

    def test_tensor_resize_numeric_control(self):
        frames = np.random.default_rng(6).integers(0, 256, (5, 128, 224, 3), dtype=np.uint8)
        result = prepare_rgb8(frames, 256, 448, tensor_resize=True)
        self.assertLessEqual(result.resize_control["maxRgb8Difference"], 2)


if __name__ == "__main__":
    unittest.main()
