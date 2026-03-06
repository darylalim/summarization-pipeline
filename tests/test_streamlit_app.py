from unittest.mock import MagicMock, patch

import torch

from streamlit_app import chunk, extract, get_device, summarize


def _make_encoded(input_ids: torch.Tensor) -> MagicMock:
    attention_mask = torch.ones_like(input_ids)
    encoded = MagicMock()
    encoded.__getitem__ = lambda self, key: {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }[key]
    encoded.keys.return_value = ["input_ids", "attention_mask"]
    encoded.__iter__ = lambda self: iter(["input_ids", "attention_mask"])
    encoded.to.return_value = encoded
    return encoded


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


class TestExtract:
    @patch("streamlit_app.Article")
    def test_returns_article(self, mock_article_cls: MagicMock) -> None:
        mock_article = MagicMock()
        mock_article.title = "Breaking News"
        mock_article.authors = ["Jane Doe"]
        mock_article.publish_date = "2026-01-15"
        mock_article.text = "Article body text."
        mock_article_cls.return_value = mock_article

        result = extract("https://example.com/article")

        mock_article_cls.assert_called_once_with("https://example.com/article")
        mock_article.download.assert_called_once()
        mock_article.parse.assert_called_once()
        mock_article.nlp.assert_called_once()
        assert result is mock_article


class TestChunk:
    def test_short_text_single_chunk(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(100))

        result = chunk("Short article text.", tokenizer)

        assert result == ["Short article text."]
        tokenizer.encode.assert_called_once_with(
            "Short article text.", add_special_tokens=False
        )

    def test_long_text_splits_into_chunks(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(2048))
        tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        result = chunk("Long article text.", tokenizer)

        assert result == ["chunk one text", "chunk two text"]
        assert tokenizer.decode.call_count == 2
        tokenizer.decode.assert_any_call(
            list(range(1024)), skip_special_tokens=True
        )
        tokenizer.decode.assert_any_call(
            list(range(1024, 2048)), skip_special_tokens=True
        )

    def test_empty_text_returns_empty(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []

        result = chunk("", tokenizer)

        assert result == []


class TestSummarize:
    def test_returns_response_and_counts(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        output_ids = torch.tensor([[1, 2, 3]])

        encoded = _make_encoded(input_ids)

        tokenizer = MagicMock()
        tokenizer.return_value = encoded
        tokenizer.decode.return_value = "A short summary."

        model = MagicMock()
        model.generate.return_value = output_ids

        response, prompt_eval_count, eval_count = summarize(
            ["Some long document text."], model, tokenizer, "cpu"
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

    def test_multi_chunk_concatenates(self) -> None:
        input_ids_1 = torch.tensor([[1, 2, 3]])
        input_ids_2 = torch.tensor([[4, 5]])
        output_ids_1 = torch.tensor([[10, 11]])
        output_ids_2 = torch.tensor([[12, 13, 14]])

        encoded_1 = _make_encoded(input_ids_1)
        encoded_2 = _make_encoded(input_ids_2)

        tokenizer = MagicMock()
        tokenizer.side_effect = [encoded_1, encoded_2]
        tokenizer.decode.side_effect = ["Summary one.", "Summary two."]

        model = MagicMock()
        model.generate.side_effect = [output_ids_1, output_ids_2]

        response, prompt_eval_count, eval_count = summarize(
            ["Chunk one text.", "Chunk two text."], model, tokenizer, "cpu"
        )

        assert response == "Summary one. Summary two."
        assert prompt_eval_count == 5  # 3 + 2
        assert eval_count == 5  # 2 + 3
        assert tokenizer.call_args_list == [
            (
                ("Chunk one text.",),
                {"return_tensors": "pt", "truncation": True, "max_length": 1024},
            ),
            (
                ("Chunk two text.",),
                {"return_tensors": "pt", "truncation": True, "max_length": 1024},
            ),
        ]

    def test_empty_chunks(self) -> None:
        model = MagicMock()
        tokenizer = MagicMock()

        response, prompt_eval_count, eval_count = summarize([], model, tokenizer, "cpu")

        assert response == ""
        assert prompt_eval_count == 0
        assert eval_count == 0
        tokenizer.assert_not_called()
        model.generate.assert_not_called()
