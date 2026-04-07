import csv
import io
from unittest.mock import MagicMock, patch

import pytest

from streamlit_app import (
    DEFAULT_GENERATION_PARAMS,
    MAX_CHUNK_TOKENS,
    SUMMARIZE_PROMPT,
    chunk,
    collection_to_csv,
    extract,
    summarize,
)


class TestDefaultGenerationParams:
    def test_has_expected_keys(self) -> None:
        expected_keys = {
            "max_tokens",
            "temp",
            "top_p",
            "repetition_penalty",
        }
        assert set(DEFAULT_GENERATION_PARAMS.keys()) == expected_keys

    def test_has_expected_values(self) -> None:
        assert DEFAULT_GENERATION_PARAMS["max_tokens"] == 256
        assert DEFAULT_GENERATION_PARAMS["temp"] == 0.0
        assert DEFAULT_GENERATION_PARAMS["top_p"] == 1.0
        assert DEFAULT_GENERATION_PARAMS["repetition_penalty"] == 1.2


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
        tokenizer.encode.return_value = list(range(MAX_CHUNK_TOKENS * 2))
        tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        result = chunk("Long article text.", tokenizer)

        assert result == ["chunk one text", "chunk two text"]
        assert tokenizer.decode.call_count == 2
        tokenizer.decode.assert_any_call(
            list(range(MAX_CHUNK_TOKENS)), skip_special_tokens=True
        )
        tokenizer.decode.assert_any_call(
            list(range(MAX_CHUNK_TOKENS, MAX_CHUNK_TOKENS * 2)),
            skip_special_tokens=True,
        )

    def test_exact_max_tokens_single_chunk(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_CHUNK_TOKENS))

        result = chunk("Exactly max tokens.", tokenizer)

        assert result == ["Exactly max tokens."]
        tokenizer.decode.assert_not_called()

    def test_max_plus_one_tokens_splits(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(MAX_CHUNK_TOKENS + 1))
        tokenizer.decode.side_effect = ["chunk one text", "chunk two text"]

        result = chunk("Just over max tokens.", tokenizer)

        assert result == ["chunk one text", "chunk two text"]
        assert tokenizer.decode.call_count == 2
        tokenizer.decode.assert_any_call(
            list(range(MAX_CHUNK_TOKENS)), skip_special_tokens=True
        )
        tokenizer.decode.assert_any_call([MAX_CHUNK_TOKENS], skip_special_tokens=True)

    def test_empty_text_returns_empty(self) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []

        result = chunk("", tokenizer)

        assert result == []


class TestSummarize:
    @patch("streamlit_app.generate")
    def test_returns_response_and_counts(self, mock_generate: MagicMock) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.encode.side_effect = [
            [1, 2, 3, 4, 5],  # prompt tokens
            [1, 2, 3],  # output tokens
        ]
        mock_generate.return_value = "A short summary."
        model = MagicMock()

        response, prompt_eval_count, eval_count = summarize(
            ["Some long document text."], model, tokenizer
        )

        assert response == "A short summary."
        assert prompt_eval_count == 5
        assert eval_count == 3
        tokenizer.apply_chat_template.assert_called_once_with(
            [
                {
                    "role": "user",
                    "content": f"{SUMMARIZE_PROMPT}Some long document text.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        mock_generate.assert_called_once_with(
            model,
            tokenizer,
            prompt="formatted prompt",
            **DEFAULT_GENERATION_PARAMS,
        )

    @patch("streamlit_app.generate")
    def test_multi_chunk_concatenates(self, mock_generate: MagicMock) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = ["prompt one", "prompt two"]
        tokenizer.encode.side_effect = [
            [1, 2, 3],  # prompt 1 tokens
            [1, 2],  # output 1 tokens
            [4, 5],  # prompt 2 tokens
            [1, 2, 3],  # output 2 tokens
        ]
        mock_generate.side_effect = ["Summary one.", "Summary two."]
        model = MagicMock()

        response, prompt_eval_count, eval_count = summarize(
            ["Chunk one text.", "Chunk two text."], model, tokenizer
        )

        assert response == "Summary one. Summary two."
        assert prompt_eval_count == 5  # 3 + 2
        assert eval_count == 5  # 2 + 3
        assert mock_generate.call_count == 2
        assert tokenizer.apply_chat_template.call_count == 2

    @patch("streamlit_app.generate")
    def test_custom_generation_params(self, mock_generate: MagicMock) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted prompt"
        tokenizer.encode.side_effect = [
            [1, 2, 3],  # prompt tokens
            [1, 2],  # output tokens
        ]
        mock_generate.return_value = "Custom summary."
        model = MagicMock()

        custom_params: dict[str, int | float] = {
            "max_tokens": 512,
            "temp": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.5,
        }

        response, prompt_eval_count, eval_count = summarize(
            ["Some text."], model, tokenizer, custom_params
        )

        assert response == "Custom summary."
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temp"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["repetition_penalty"] == 1.5

    @patch("streamlit_app.generate")
    def test_empty_chunks(self, mock_generate: MagicMock) -> None:
        model = MagicMock()
        tokenizer = MagicMock()

        response, prompt_eval_count, eval_count = summarize([], model, tokenizer)

        assert response == ""
        assert prompt_eval_count == 0
        assert eval_count == 0
        tokenizer.apply_chat_template.assert_not_called()
        mock_generate.assert_not_called()


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

        assert rows[0]["max_tokens"] == "256"
        assert rows[0]["temp"] == "0.0"
        assert rows[0]["top_p"] == "1.0"
        assert rows[0]["repetition_penalty"] == "1.2"
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
