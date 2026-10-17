"""Scalar/configuration contracts only; no CPU CNN or GPU execution."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_image_inputs as images
import cnn_graph_acceptance as graphs


class ImageInputsTests(unittest.TestCase):
    def test_all_six_have_large_varied_native_images(self):
        self.assertEqual(len(images.SHAPES),6)
        self.assertGreater(len(set(images.SHAPES.values())),3)
        for height,width in images.SHAPES.values():
            self.assertGreaterEqual(min(height,width),330)
            self.assertGreaterEqual(height*width,576*330)

    def test_full_paired_targeted_matrix_keeps_learner_and_seeds_fixed(self):
        for game in images.SHAPES:
            reference,_=images.config(game,images.CONTROLS["nature"])
            for channels in (8,16):
                for residual in (0,1):
                    config,_=images.config(game,images.architecture(channels,residual))
                    for section in ("base","train","env","vec","selfplay"):
                        self.assertEqual(dict(config[section]),dict(reference[section]))
                    self.assertFalse(any(s.startswith("sweep.") for s in config.sections()))
                    self.assertEqual(config.getint("base","seed"),7)
                    self.assertEqual(config.getint("train","minibatch_size"),16)

    def test_stride_and_rgb_propagate_through_independent_scalar_topology(self):
        for stride in (1,2,4,8):
            cfg,_=images.config("flappycnn",images.architecture(stride=stride,gap=1))
            rows,count=graphs.topology(*graphs.settings(cfg),shape=(640,420,3))
            self.assertEqual(rows[0][:3],[640,420,3])
            self.assertEqual(rows[0][3:5],[(640+stride-1)//stride,(420+stride-1)//stride])
            self.assertEqual(rows[0][7],stride)
            self.assertEqual(count,8*(8*8*3+1)+64*(8+1)+128*(64+1))
            self.assertIn(3,[r[11] for r in rows])

    def test_nature_preserves_valid_convolutions_at_portrait_rgb_shape(self):
        rows,count=graphs.topology(1,[],(640,420,3))
        self.assertEqual(rows[0][:8],[640,420,3,159,104,32,8,4])
        self.assertEqual(rows[-1][2],76*49*64)
        expected=32*(8*8*3+1)+64*(4*4*32+1)+64*(3*3*64+1)+128*(76*49*64+1)
        self.assertEqual(count,expected)

    def test_invalid_stride_never_creates_a_candidate(self):
        for stride in (0,3,16):
            with self.assertRaises(ValueError): images.architecture(stride=stride)

    def test_image_case_batches_do_not_inherit_tiny_frame_batch2048(self):
        contract=dict(schema=graphs.IMAGE_SCHEMA,batches=[1,4,16])
        self.assertEqual(len(graphs.cases(contract)),9)
        self.assertNotIn((2048,"signed"),graphs.cases(contract))
        self.assertIn((2048,"signed"),graphs.cases())

    def test_preparation_marks_short_inputs_uncalibrated_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/"inputs"; result=images.prepare(out)
            self.assertEqual(len(result["entries"]),6*10)
            self.assertFalse(result["gpu_executed"])
            self.assertFalse(result["calibrated_for_learning"])
            self.assertFalse(result["seed_sweep_allowed"])
            with self.assertRaises(FileExistsError): images.prepare(out)


if __name__ == "__main__": unittest.main()
