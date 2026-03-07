import csv
import io
from unittest.mock import MagicMock, patch

import pytest
import torch

from streamlit_app import (
    DEFAULT_GENERATION_PARAMS,
    chunk,
    collection_to_csv,
    extract,
    get_device,
    summarize,
)


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


class TestDefaultGenerationParams:
    def test_has_expected_keys(self) -> None:
        expected_keys = {
            "max_length",
            "min_length",
            "num_beams",
            "do_sample",
            "length_penalty",
            "early_stopping",
            "no_repeat_ngram_size",
        }
        assert set(DEFAULT_GENERATION_PARAMS.keys()) == expected_keys

    def test_has_expected_values(self) -> None:
        assert DEFAULT_GENERATION_PARAMS["max_length"] == 130
        assert DEFAULT_GENERATION_PARAMS["min_length"] == 30
        assert DEFAULT_GENERATION_PARAMS["num_beams"] == 4
        assert DEFAULT_GENERATION_PARAMS["do_sample"] is False
        assert DEFAULT_GENERATION_PARAMS["length_penalty"] == 1.0
        assert DEFAULT_GENERATION_PARAMS["early_stopping"] is True
        assert DEFAULT_GENERATION_PARAMS["no_repeat_ngram_size"] == 3


class TestGetDevice:
    @patch("streamlit_app.torch.backends.mps.is_available", return_value=True)
    def test_mps(self, _mock_mps: MagicMock) -> None:
        assert get_device() == "mps"

    @patch("streamlit_app.torch.cuda.is_available", return_value=True)
    @patch("streamlit_app.torch.backends.mps.is_available", return_value=False)
    def test_cuda(self, _mock_cuda: MagicMock, _mock_mps: MagicMock) -> None:
        assert get_device() == "cuda"

    @patch("streamlit_app.torch.cuda.is_available", return_value=False)
    @patch("streamlit_app.torch.backends.mps.is_available", return_value=False)
    def test_cpu(self, _mock_cuda: MagicMock, _mock_mps: MagicMock) -> None:
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

    @patch("streamlit_app.Article")
    def test_download_failure_propagates(self, mock_article_cls: MagicMock) -> None:
        mock_article = MagicMock()
        mock_article.download.side_effect = Exception("Network error")
        mock_article_cls.return_value = mock_article

        with pytest.raises(Exception, match="Network error"):
            extract("https://example.com/article")


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
        tokenizer.decode.assert_any_call(list(range(1024)), skip_special_tokens=True)
        tokenizer.decode.assert_any_call(
            list(range(1024, 2048)), skip_special_tokens=True
        )

    def test_exact_1024_tokens_single_chunk(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(1024))

        result = chunk("Exactly 1024 tokens.", tokenizer)

        assert result == ["Exactly 1024 tokens."]
        tokenizer.decode.assert_not_called()

    def test_1025_tokens_splits_into_two_chunks(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(1025))
        tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        result = chunk("Just over 1024 tokens.", tokenizer)

        assert result == ["chunk one text", "chunk two text"]
        assert tokenizer.decode.call_count == 2
        tokenizer.decode.assert_any_call(list(range(1024)), skip_special_tokens=True)
        tokenizer.decode.assert_any_call([1024], skip_special_tokens=True)

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
        generate_kwargs = model.generate.call_args[1]
        for key, value in DEFAULT_GENERATION_PARAMS.items():
            assert generate_kwargs[key] == value

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
        generate_kwargs = model.generate.call_args[1]
        for key, value in DEFAULT_GENERATION_PARAMS.items():
            assert generate_kwargs[key] == value

    def test_custom_generation_params(self) -> None:
        input_ids = torch.tensor([[1, 2, 3]])
        output_ids = torch.tensor([[10, 11]])

        encoded = _make_encoded(input_ids)

        tokenizer = MagicMock()
        tokenizer.return_value = encoded
        tokenizer.decode.return_value = "Custom summary."

        model = MagicMock()
        model.generate.return_value = output_ids

        generation_params = {
            "max_length": 200,
            "min_length": 50,
            "num_beams": 2,
            "do_sample": True,
            "length_penalty": 0.5,
            "early_stopping": False,
            "no_repeat_ngram_size": 4,
        }

        response, prompt_eval_count, eval_count = summarize(
            ["Some text."], model, tokenizer, "cpu", generation_params
        )

        assert response == "Custom summary."
        generate_kwargs = model.generate.call_args[1]
        assert generate_kwargs["max_length"] == 200
        assert generate_kwargs["min_length"] == 50
        assert generate_kwargs["num_beams"] == 2
        assert generate_kwargs["do_sample"] is True
        assert generate_kwargs["length_penalty"] == 0.5
        assert generate_kwargs["early_stopping"] is False
        assert generate_kwargs["no_repeat_ngram_size"] == 4

    def test_empty_chunks(self) -> None:
        model = MagicMock()
        tokenizer = MagicMock()

        response, prompt_eval_count, eval_count = summarize([], model, tokenizer, "cpu")

        assert response == ""
        assert prompt_eval_count == 0
        assert eval_count == 0
        tokenizer.assert_not_called()
        model.generate.assert_not_called()


def _make_collection_item(
    **overrides: object,
) -> dict[str, object]:
    defaults: dict[str, object] = {
        "model": "facebook/bart-large-cnn",
        "url": "https://example.com",
        "title": "Test Article",
        "authors": ["Alice", "Bob"],
        "publish_date": "2026-01-15",
        "keywords": ["news", "test"],
        "original_text": "Original text here.",
        "response": "Summary text here.",
        "total_duration": 1.5,
        "chunk_count": 1,
        "prompt_eval_count": 100,
        "eval_count": 30,
        "original_word_count": 3,
        "summary_word_count": 3,
        "compression_ratio": 1.0,
        "generation_params": dict(DEFAULT_GENERATION_PARAMS),
    }
    defaults.update(overrides)
    return defaults


class TestCollectionToCsv:
    def test_single_item(self) -> None:
        collection = [_make_collection_item()]

        result = collection_to_csv(collection)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["title"] == "Test Article"
        assert rows[0]["authors"] == "Alice;Bob"
        assert rows[0]["keywords"] == "news;test"
        assert rows[0]["url"] == "https://example.com"

    def test_flattened_generation_params(self) -> None:
        collection = [_make_collection_item()]

        result = collection_to_csv(collection)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert rows[0]["max_length"] == "130"
        assert rows[0]["min_length"] == "30"
        assert rows[0]["num_beams"] == "4"
        assert rows[0]["do_sample"] == "False"
        assert rows[0]["length_penalty"] == "1.0"
        assert rows[0]["early_stopping"] == "True"
        assert rows[0]["no_repeat_ngram_size"] == "3"
        assert "generation_params" not in rows[0]

    def test_multi_item(self) -> None:
        collection = [
            _make_collection_item(title="First Article"),
            _make_collection_item(title="Second Article", url="https://example.com/2"),
        ]

        result = collection_to_csv(collection)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["title"] == "First Article"
        assert rows[1]["title"] == "Second Article"
        assert rows[1]["url"] == "https://example.com/2"

    def test_empty_authors_and_keywords(self) -> None:
        collection = [_make_collection_item(authors=[], keywords=[])]

        result = collection_to_csv(collection)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert rows[0]["authors"] == ""
        assert rows[0]["keywords"] == ""

    def test_special_characters(self) -> None:
        collection = [
            _make_collection_item(
                original_text='He said, "hello"\nNew line here.',
                response="Commas, quotes, and\nnewlines.",
            )
        ]

        result = collection_to_csv(collection)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert rows[0]["original_text"] == 'He said, "hello"\nNew line here.'
        assert rows[0]["response"] == "Commas, quotes, and\nnewlines."

    def test_excludes_id_field(self) -> None:
        collection = [_make_collection_item(_id="test-uuid-123")]

        result = collection_to_csv(collection)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)

        assert "_id" not in rows[0]

    def test_empty_collection(self) -> None:
        result = collection_to_csv([])
        assert result == ""
