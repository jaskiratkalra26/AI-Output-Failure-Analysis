# Configuration Guide

This guide explains the parameters in `config/config.py`. You can adjust these settings to tune the performance, sensitivity, and cost of the Hallucination Detector.

## 1. Model Configuration (`model_config`)
Controls the AI models used for claim extraction and verification.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gemini_model` | `gemini-2.5-flash` | The specific Google Gemini model used for extracting atomic claims. Flash models are faster and cheaper. |
| `nli_model_name` | `roberta-large-mnli` | The Hugging Face model used for claim verification. `roberta-large-mnli` is robust for Natural Language Inference. |

## 2. Retrieval Settings (`retrieval`)
Controls how evidence is gathered from Wikipedia and Web Search.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_search_results` | `1` | Number of top Google Search results to parse per claim (via SerpAPI). |
| `max_paragraphs` | `100` | Maximum paragraphs to process from a single Wikipedia article to avoid context overflow. |
| `min_sentence_length` | `20` | Ignore sentences shorter than this (in characters) during evidence gathering. |
| `document_top_n_sentences`| `5` | The number of most relevant sentences to keep from retrieved documents for final verification. |
| `fallback_threshold_score`| `2` | If the relevance score of retrieved evidence is below this, the system tries SerpAPI fallback (if configured). |

## 3. Verification Thresholds (`verification`)
These are the most critical settings for defining what counts as a hallucination.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contradiction_threshold` | `0.80` | **Sensitivity control.** If the NLI model predicts "contradiction" with probability > 80%, the claim is marked as **Refuted**. Lowering this makes the system stricter. |
| `keyword_overlap_threshold`| `0.7` | Used for simple heuristic checks if deep learning verification is ambiguous. |
| `supportive_entailment_confidence` | `0.9` | If the NLI model predicts "entailment" with > 90% confidence, the claim is marked as **Supported**. |

## 4. Extraction Settings (`extraction`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `naive_fallback_min_length` | `10` | If the LLM fails to extract claims, the system splits text by periods. Segments shorter than this are ignored. |
