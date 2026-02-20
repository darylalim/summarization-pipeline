from unittest.mock import MagicMock, patch

import torch

from streamlit_app import convert, get_device, summarize


class TestGetDevice:
    @patch("streamlit_app.torch.backends.mps.is_available", return_value=True)
    def test_mps(self, _mock_mps: MagicMock) -> None:
        assert get_device() == "mps"

    @patch("streamlit_app.torch.cuda.is_available", return_value=True)
    @patch("streamlit_app.torch.backends.mps.is_available", return_value=False)
    def test_cuda(self, _mock_mps: MagicMock, _mock_cuda: MagicMock) -> None:
        assert get_device() == "cuda"

    @patch("streamlit_app.torch.cuda.is_available", return_value=False)
    @patch("streamlit_app.torch.backends.mps.is_available", return_value=False)
    def test_cpu(self, _mock_mps: MagicMock, _mock_cuda: MagicMock) -> None:
        assert get_device() == "cpu"


class TestConvert:
    def test_returns_markdown(self) -> None:
        doc_converter = MagicMock()
        doc_converter.convert.return_value.document.export_to_markdown.return_value = (
            "# Title\n\nBody text."
        )

        result = convert("/tmp/test.pdf", doc_converter)

        assert result == "# Title\n\nBody text."
        doc_converter.convert.assert_called_once_with(
            source="/tmp/test.pdf",
            max_num_pages=100,
            max_file_size=20 * 1024 * 1024,
        )


class TestSummarize:
    def test_returns_response_and_counts(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)
        output_ids = torch.tensor([[1, 2, 3]])

        encoded = MagicMock()
        encoded.__getitem__ = lambda self, key: {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }[key]
        encoded.keys.return_value = ["input_ids", "attention_mask"]
        encoded.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
        encoded.to.return_value = encoded

        tokenizer = MagicMock()
        tokenizer.return_value = encoded
        tokenizer.decode.return_value = "A short summary."

        model = MagicMock()
        model.generate.return_value = output_ids

        response, prompt_eval_count, eval_count = summarize(
            "Some long document text.", model, tokenizer, "cpu"
        )

        assert response == "A short summary."
        assert prompt_eval_count == 5
        assert eval_count == 3
        tokenizer.assert_called_once_with(
            "Some long document text.",
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        encoded.to.assert_called_once_with("cpu")
        tokenizer.decode.assert_called_once()
        decode_args, decode_kwargs = tokenizer.decode.call_args
        assert torch.equal(decode_args[0], output_ids[0])
        assert decode_kwargs == {"skip_special_tokens": True}
