import contextlib, io, unittest, tempfile, torch, json, os,  numpy as np

from unittest.mock import MagicMock, patch
from training.distillation import DEFAULT_EVENT_DENSITY_REFERENCE, event_density, get_event_density_reference

from pathlib import Path
from unittest.mock import patch

from tools.measure_event_density import get_event_density_dict, main


class MeasureEventDensityTests(unittest.TestCase):

    @patch('tools.measure_event_density.event_coverage')
    @patch('tools.measure_event_density.load_causal_events')
    @patch('tools.measure_event_density.yaml_file_to_dict')
    @patch('tools.measure_event_density.DSECFusionDataset')
    def test_normalised_density_calculation(self, dataset_class, yaml_mock, load_events_mock, coverage_mock):
        dataset = dataset_class.return_value
        base = dataset.base_dataset
        test_path = Path(__file__).resolve().parents[1] / 'docs' / 'output.json'

        base.__len__.return_value = 5
        base.height = 16
        base.width = 16
        base.subsequence_directories = [Path('seq_a')]

        yaml_mock.return_value = {'train': ['seq_a']}

        # Gives each frame a known score
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        coverage_mock.side_effect = [torch.tensor(score) for score in scores]

        # Calculates percentiles w/ dummy output path
        percentiles = get_event_density_dict(dataset, test_path)

        # Assert all frames are called
        self.assertEqual(load_events_mock.call_count, 5)
        dataset.generate_event_histogram.assert_called()
        self.assertEqual(dataset.generate_event_histogram.call_count, 5)
        self.assertEqual(coverage_mock.call_count, 5)

        # Adds dummy output file to cleanup; file will be destroyed when test finishes
        self.addCleanup(test_path.unlink, missing_ok=True)

        # Assert the percentiles are correct
        self.assertEqual(percentiles, {'10': 0.14, '25': 0.2, '50': 0.3, '75': 0.4, '90': 0.46})
        output_js_file = open(test_path, mode='r')
        output_js = json.load(output_js_file)
        self.assertEqual(output_js.get('percentiles'), percentiles)
        self.assertEqual(output_js.get('event_density_reference'), percentiles.get('50'))


    def test_raise_SystemExit_error_if_json_file_exists(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.json"

            # Create a dummy JSON file beforehand
            original_metadata = {"reference": 0.1234}
            output_path.write_text(
                json.dumps(original_metadata),
                encoding="utf-8",
            )
            # Check that the file was created
            self.assertTrue(output_path.exists())

            cli_args = ['measure_event_density.py', '--dataset-path', 'fake/path', '--output', str(output_path),]
            with patch("sys.argv", cli_args):
                with self.assertRaises(SystemExit) as context:
                    main()

            self.assertEqual(context.exception.code, 2)


    @patch('tools.measure_event_density.yaml_file_to_dict')
    @patch('tools.measure_event_density.DSECFusionDataset')
    def test_missing_sequences_raises_error(self, dataset_class, yaml_mock):
        dataset = dataset_class.return_value
        dataset.base_dataset = MagicMock()
        base = dataset.base_dataset
        base.__len__.return_value = 1
        test_path = Path(__file__).resolve().parents[1] / 'docs' / 'output.json'

        base.subsequence_directories = [Path('seq_a')]
        yaml_mock.return_value = {'train': ['seq_a', 'seq_b']}

        # Calculates percentiles w/ dummy output path
        with self.assertRaisesRegex(ValueError, 'Complete training split required; missing sequences: seq_b') as context:
            percentiles = get_event_density_dict(dataset, test_path)

    @patch('tools.measure_event_density.json.load')
    @patch('tools.measure_event_density.open', create=True)
    def test_replace_event_density_reference_if_json_exists(self, mock_open, mock_json_load):
        mock_json_load.return_value = {'event_density_reference': 0.5}
        reference = get_event_density_reference()
        self.assertEqual(reference, 0.5)


    def test_keep_event_density_reference_if_json_does_not_exist(self):
        reference = get_event_density_reference()
        self.assertEqual(reference, DEFAULT_EVENT_DENSITY_REFERENCE)

        coverage = torch.tensor(DEFAULT_EVENT_DENSITY_REFERENCE/2)
        self.assertEqual(event_density(coverage), torch.tensor(0.5))

    @patch('tools.measure_event_density.json.load')
    @patch('tools.measure_event_density.open', create=True)
    def test_keep_event_density_reference_if_json_exists_without_reference(self, mock_open, mock_json_load):
        mock_json_load.return_value = {'not_event_density_reference': 0.5}
        reference = get_event_density_reference()
        self.assertEqual(reference, DEFAULT_EVENT_DENSITY_REFERENCE)


    def test_event_density_clipping(self):
        coverage = torch.tensor([[-0.4, DEFAULT_EVENT_DENSITY_REFERENCE/4, 0.5], [1.0, 1.5, 2.0]])

        equality = torch.equal(event_density(coverage), torch.tensor([[0, 0.25, 1], [1, 1, 1]]))
        self.assertTrue(equality)

