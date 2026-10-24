"""Offline regressions for prediction extraction and open-ended QA judging."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


# Importing the evaluator also imports optional video decoding and Azure clients.
# Neither is used by these tests; keep imports independent of media/credentials.
provider = types.ModuleType("tools.gpt_api")
provider.azure_gpt4_client = None
import_stubs = {"tools.gpt_api": provider}
if importlib.util.find_spec("decord") is None:
    import_stubs["decord"] = types.ModuleType("decord")
previous_modules = {name: sys.modules.get(name) for name in import_stubs}
sys.modules.update(import_stubs)
try:
    from evaluation.evaluate import read_dataset
    from evaluation.metrics import evaluate_qa_oe_gpt as qa
finally:
    for name, previous in previous_modules.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class OpenEndedQATest(unittest.TestCase):
    def read_prediction(self, directory, dataset):
        row = {
            "dataset": dataset,
            "idx": 0,
            "messages": [
                {"role": "user", "content": [
                    {"type": "video", "video": {"video_file": "unused.mp4"}},
                    {"type": "text", "text": "What is the person doing?"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Running", "reference": "RUNNING"},
                ]},
            ],
        }
        path = Path(directory) / "predictions.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return read_dataset(str(path), dataset)[0]

    def test_extracted_predictions_are_judged_and_summarized(self):
        for dataset in ("MSVD-QA-val", "MSR-VTT-QA-val", "TGIF-QA-test", "ActivityNet-QA-test"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                item = self.read_prediction(directory, dataset)
                self.assertNotIn("question", item)
                with patch.object(qa, "call_azure_gpt_api", return_value="{'pred': 'yes', 'score': 5}") as judge:
                    result = qa.process_one_sample((item, "test-model", False))
                judge.assert_called_once_with("What is the person doing?", "running", "running", "test-model")
                self.assertTrue(result["success"])
                self.assertIs(result["data"], item)
                metric = qa.GPTMetric(dataset)
                metric.update_metric(result)
                with contextlib.redirect_stdout(io.StringIO()):
                    metric.summarize_metric()
                metric.save_results(directory)
                saved = (Path(directory) / "eval_records" / f"{dataset}_eval_result.txt").read_text()
                self.assertIn("#Successful Results: 1\n#Failed Results: 0", saved)
                self.assertIn("Accuracy: 100.0", saved)
                self.assertIn("Average Score: 5.0", saved)

    def test_invalid_judge_responses_remain_failed_results(self):
        item = {"prompt": "Question?", "response": "Answer", "prediction": "Prediction"}
        for response in ("not a dictionary", "{'pred': 'maybe', 'score': 3}",
                         "{'pred': 'yes', 'score': -1}", "{'pred': 'yes', 'score': 6}"):
            with self.subTest(response=response):
                with patch.object(qa, "call_azure_gpt_api", return_value=response):
                    result = qa.process_one_sample((item, "test-model", False))
                self.assertFalse(result["success"])
                metric = qa.GPTMetric("MSVD-QA-val")
                metric.update_metric(result)
                self.assertEqual(metric.results, [])
                self.assertEqual(metric.invalid_results, [result])

    def test_negative_judgment_and_provider_failure(self):
        item = {"prompt": "Question?", "response": "Answer", "prediction": "Prediction"}
        with patch.object(qa, "call_azure_gpt_api", return_value="{'pred': 'no', 'score': 0}"):
            result = qa.process_one_sample((item, "test-model", False))
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], {"pred": "no", "score": 0.0})
        with patch.object(qa, "call_azure_gpt_api", side_effect=ValueError("invalid deployment")) as judge:
            result = qa.process_one_sample((item, "test-model", False))
        judge.assert_called_once()
        self.assertFalse(result["success"])
        self.assertEqual(result["result"], "invalid deployment")


if __name__ == "__main__":
    unittest.main()
