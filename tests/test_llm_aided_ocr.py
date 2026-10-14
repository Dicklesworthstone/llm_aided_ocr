import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_aided_ocr  # noqa: E402

RAW_OCR_PATH = REPO_ROOT / "160301289-Warren-Buffett-Katharine-Graham-Letter__raw_ocr_output.txt"
CORRECTED_MD_PATH = REPO_ROOT / "160301289-Warren-Buffett-Katharine-Graham-Letter_llm_corrected.md"
SAMPLE_PDF_PATH = REPO_ROOT / "160301289-Warren-Buffett-Katharine-Graham-Letter.pdf"
CLI_PATH = REPO_ROOT / "llm-aided-ocr-cli.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("llm_aided_ocr_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b1_api_key_validation_and_lazy_clients():
    assert not llm_aided_ocr.is_valid_api_key(None)
    assert not llm_aided_ocr.is_valid_api_key("")
    assert not llm_aided_ocr.is_valid_api_key("   ")
    assert not llm_aided_ocr.is_valid_api_key("your-openai-api-key")
    assert not llm_aided_ocr.is_valid_api_key("your-anthropic-api-key")
    assert not llm_aided_ocr.is_valid_api_key("your-custom-key")
    assert not llm_aided_ocr.is_valid_api_key("changeme")
    assert llm_aided_ocr.is_valid_api_key("sk-proj-valid-looking-test-key-12345")

    client = llm_aided_ocr.get_openai_client("sk-test-lazy-init")
    assert client is not None
    assert llm_aided_ocr.get_openai_client("sk-test-lazy-init") is client

    gpu_info = llm_aided_ocr.is_gpu_available()
    assert isinstance(gpu_info, dict)
    assert "gpu_found" in gpu_info
    assert "num_gpus" in gpu_info
    assert "total_vram" in gpu_info


def test_b2_tokenizer_caching_and_unknown_model_fallback():
    t1 = llm_aided_ocr.get_tokenizer("gpt-4o-mini")
    t2 = llm_aided_ocr.get_tokenizer("gpt-4o-mini")
    assert t1 is t2

    for model in ("claude-3-haiku-20240307", "llama-3.1-8b", "o3-mini", "unknown-future-model-v9"):
        tok = llm_aided_ocr.get_tokenizer(model)
        encoded = tok.encode("Pension liabilities compound over time.")
        assert len(encoded) >= 4
        assert llm_aided_ocr.estimate_tokens("Pension liabilities compound over time.", model) == len(encoded)


def test_b2_chunk_text_no_empty_first_chunk_and_long_sentence_splitting():
    # A single sentence much longer than max_chunk_tokens=20
    long_sentence = " ".join(f"word{i}" for i in range(1, 85)) + "."
    chunks = llm_aided_ocr.chunk_text(long_sentence, max_chunk_tokens=20, model_name="gpt-4o-mini", overlap_size=0)
    assert len(chunks) >= 4
    assert all(c.strip() != "" for c in chunks)
    reconstructed_words = " ".join(chunks).split()
    assert reconstructed_words == long_sentence.split()


def test_b2_adjust_overlaps_never_deletes_chunk_words():
    tokenizer = llm_aided_ocr.get_tokenizer("gpt-4o-mini")
    chunk_a = " ".join(f"alpha{i}" for i in range(1, 60))
    chunk_b = " ".join(f"beta{i}" for i in range(1, 60))

    # Set max_chunk_tokens just barely equal to chunk_b's size so overlap cannot fit in full
    b_size = len(tokenizer.encode(chunk_b))
    adjusted = llm_aided_ocr.adjust_overlaps([chunk_a, chunk_b], tokenizer, max_chunk_tokens=b_size, overlap_size=50)
    assert len(adjusted) == 2
    # Every single word of chunk_b MUST still be present (the old code deleted the last 50 words of chunk_b!)
    for word in chunk_b.split():
        assert word in adjusted[1].split()


def test_b2_split_oversized_prompt_preserves_instructions():
    body = " ".join(f"Sentence number {i} has OCR errors." for i in range(1, 120))
    prompt = (
        "Correct OCR-induced errors in the text.\n\n"
        "Current chunk to process:\n"
        f"{body}\n\n"
        "Corrected text:\n"
    )
    sub_prompts = llm_aided_ocr.split_oversized_prompt(prompt, max_prompt_tokens=150, model_name="gpt-4o-mini")
    assert len(sub_prompts) >= 3
    for sp in sub_prompts:
        assert "Correct OCR-induced errors in the text." in sp
        assert "Current chunk to process:\n" in sp
        assert "Corrected text:" in sp


def test_b3_preprocess_image_and_deskew_real_pixel_transform():
    # Create a synthetic low-contrast gray image with dark horizontal text bars tilted by 4 degrees
    canvas = Image.new("RGB", (600, 400), color=(210, 210, 210))
    draw = ImageDraw.Draw(canvas)
    for y in range(80, 320, 35):
        draw.rectangle([80, y, 520, y + 10], fill=(70, 70, 70))
    tilted = canvas.rotate(4.0, resample=Image.Resampling.BICUBIC, fillcolor=(210, 210, 210))

    processed = llm_aided_ocr.preprocess_image(tilted, deskew=True, apply_clahe=True, binarize=True)
    arr = np.array(processed)
    assert arr.ndim == 2
    assert arr.shape == (400, 600)
    # Binarized Otsu output contains only 0 and 255
    unique_vals = set(np.unique(arr).tolist())
    assert unique_vals.issubset({0, 255})
    assert 0 in unique_vals and 255 in unique_vals

    # Verify deskew_image reduces the measured skew angle on the tilted grayscale image
    raw_gray = cv2.cvtColor(np.array(tilted), cv2.COLOR_RGB2GRAY)
    deskewed = llm_aided_ocr.deskew_image(raw_gray)
    assert not np.array_equal(raw_gray, deskewed)


def test_b3_real_pdf_conversion_and_ocr_on_buffett_pdf():
    assert SAMPLE_PDF_PATH.exists()
    images = llm_aided_ocr.convert_pdf_to_images(str(SAMPLE_PDF_PATH), max_pages=1, skip_first_n_pages=0, dpi=200)
    assert len(images) == 1
    ocr_text = llm_aided_ocr.ocr_image(images[0])
    assert "PENSIONS" in ocr_text
    assert "Irreversible Nature of Pension Promises" in ocr_text


def test_b4_page_number_stripping_and_footnote_relocation_on_buffett_raw_ocr():
    raw_text = RAW_OCR_PATH.read_text(encoding="utf-8")
    pages = [p for p in raw_text.split("\x0c") if p.strip()]
    assert len(pages) == 19

    # Verify OCR-mangled page number detector catches all mangled page tokens in the Buffett letter
    for mangled in ("~2-", "-y", "-6-", "-7-", "-~8-", "-9-", "-1O-", "-LL-", "-12-", "-13-", "~Li-", "~L5-", "-l6-", "-17-", "-18-", "~19-"):
        assert llm_aided_ocr.is_ocr_page_number_line(mangled), f"Failed to detect page token: {mangled}"

    # Verify non-page tokens are never matched
    for valid_line in ("-0.3%", "(0.3%)", "$70,000.", "* My views regarding inflationary possibilities", "- Dow-Jones Industrials +2.8%"):
        assert not llm_aided_ocr.is_ocr_page_number_line(valid_line)

    cleaned_pages = llm_aided_ocr.clean_page_artifacts(pages, suppress_headers_and_page_numbers=True)
    cleaned_full = "\n\n".join(cleaned_pages)

    # None of the standalone mangled page tokens should survive in cleaned_full
    for mangled in ("~2-", "-1O-", "-LL-", "~Li-", "~L5-", "-l6-", "~19-"):
        assert f"\n{mangled}\n" not in f"\n{cleaned_full}\n"

    # All 4 mid-sentence page breaks interrupted by bottom-of-page footnotes must now be healed into contiguous sentences!
    assert "reduced your net worth today by about $70,000." in cleaned_full
    assert "It is difficult to measure turnover costs with precision" in cleaned_full
    assert "usually comes down to the past record, particularly the recent past" in cleaned_full
    assert "And occasionally there is an offsetting advantage" in cleaned_full

    # And all 3 previously deleted footnotes must be 100% present in cleaned_full!
    assert "My views regarding inflationary possibilities are more extreme" in cleaned_full
    assert "An interesting example of this line of thinking (sub-species:" in cleaned_full
    assert "I am leaving most of the legal discussion out of this memo" in cleaned_full


@pytest.mark.asyncio
async def test_b5_retry_with_backoff_and_bounded_concurrency():
    attempts = 0

    async def flaky_op():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("Transient 429 rate limit")
        return "recovered-ok"

    result = await llm_aided_ocr.retry_with_backoff(flaky_op, max_retries=3, base_delay=0.01, operation_name="test_op")
    assert result == "recovered-ok"
    assert attempts == 3

    # Verify bounded concurrency in process_chunks
    active_count = 0
    peak_active = 0

    async def fake_process_chunk(chunk, prev_context, idx, total, reformat, suppress):
        nonlocal active_count, peak_active
        active_count += 1
        peak_active = max(peak_active, active_count)
        await asyncio.sleep(0.02)
        active_count -= 1
        return f"processed-{idx}:{chunk}", chunk[-20:]

    with (
        patch.object(llm_aided_ocr, "USE_LOCAL_LLM", False),
        patch.object(llm_aided_ocr, "MAX_CONCURRENT_REQUESTS", 2),
        patch.object(llm_aided_ocr, "process_chunk", side_effect=fake_process_chunk),
    ):
        chunks = [f"chunk-{i}" for i in range(6)]
        outputs = await llm_aided_ocr.process_chunks(chunks, reformat_as_markdown=True, suppress_headers_and_page_numbers=True)
        assert outputs == [f"processed-{i}:chunk-{i}" for i in range(6)]
        assert peak_active <= 2


@pytest.mark.asyncio
async def test_b6_local_llm_model_singleton_cache(tmp_path):
    llm_aided_ocr.clear_model_cache()
    models_dir = Path(llm_aided_ocr.__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    dummy_model_path = models_dir / "test_dummy_cache_model.gguf"
    created_dummy = False
    if not dummy_model_path.exists():
        dummy_model_path.write_bytes(b"GGUF-test")
        created_dummy = True

    load_calls = 0

    class DummyLlama:
        def __init__(self, **kwargs):
            nonlocal load_calls
            load_calls += 1
            self.kwargs = kwargs

        def __call__(self, prompt, max_tokens=100, temperature=0.7, **kwargs):
            return {
                "choices": [{"text": f"Corrected: {prompt[:20]}", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5, "total_time": 0.01},
            }

    try:
        with patch.object(llm_aided_ocr, "Llama", DummyLlama):
            m1 = llm_aided_ocr.load_model("test_dummy_cache_model.gguf")
            m2 = llm_aided_ocr.load_model("test_dummy_cache_model.gguf")
            assert m1 is m2
            assert load_calls == 1

            out = await llm_aided_ocr.generate_completion_from_local_llm(
                "test_dummy_cache_model.gguf",
                "Short test prompt",
                number_of_tokens_to_generate=32,
            )
            assert out is not None and out.startswith("Corrected:")
            assert load_calls == 1
    finally:
        llm_aided_ocr.clear_model_cache()
        if created_dummy and dummy_model_path.exists():
            dummy_model_path.unlink()


def test_b7_stitch_chunks_repairs_all_five_buffett_seams():
    # Seam 1: mid-sentence split with echoed heading and duplicated phrase
    seam1 = llm_aided_ocr.stitch_chunks(
        [
            "In Germany, the entire Daimler Benz Company was selling in the market for the price of 300 motor cars. Almost all past investments",
            "# the price of 300 motor cars.\n\nAlmost all past investments were nearly worthless, and current salary levels were astronomical.",
        ]
    )
    assert "# the price" not in seam1
    assert seam1.count("Almost all past investments") == 1
    assert "for the price of 300 motor cars. Almost all past investments were nearly worthless" in seam1

    # Seam 2: echoed 10-word suffix at start of next chunk
    seam2 = llm_aided_ocr.stitch_chunks(
        [
            "(If interest rates go up, bond prices must go down—and if the bonds are long-term and the rates rise sharply, prices go into a power dive.)",
            "the rates rise sharply, prices go into a power dive.)\n\n4. Many managements thus saw their largest division earning dismal rates.",
        ]
    )
    assert seam2.count("prices go into a power dive") == 1
    assert ")the rates" not in seam2

    # Seam 3: echoed heading from preceding sentence
    seam3 = llm_aided_ocr.stitch_chunks(
        [
            "Their bare statistical record will not be distinguishable from the larger group who have been calling flips indiscriminately and have been lucky - so far.",
            "# Calling Flips Indiscriminately\n\nIt may be possible, if you know a good deal about investments, to talk with a manager.",
        ]
    )
    assert "# Calling Flips Indiscriminately" not in seam3
    assert "lucky - so far.\n\nIt may be possible" in seam3

    # Seam 4: echoed sentence turned into heading
    seam4 = llm_aided_ocr.stitch_chunks(
        [
            "The reason isn't too hard to see.",
            "# reason isn't too hard to see (particularly with a rear view mirror).\n\nMuch of Wall Street is a succession of fashions.",
        ]
    )
    assert "# reason" not in seam4
    assert "The reason isn't too hard to see (particularly with a rear view mirror)." in seam4

    # Seam 5: echoed heading + echoed fragment
    seam5 = llm_aided_ocr.stitch_chunks(
        [
            "A second advantage would be flexibility, allowing portions of business operations to be acquired or divested much more easily than entire operating businesses.",
            "# Acquired or Divested\n\nMuch more easily than entire operating businesses. It must be emphasized that measurement of this program would be based on earnings.",
        ]
    )
    assert "# Acquired or Divested" not in seam5
    assert seam5.lower().count("much more easily than entire operating businesses") == 1


def test_b7_and_b8_buffett_flagship_sample_conservation():
    raw_text = RAW_OCR_PATH.read_text(encoding="utf-8")
    md_text = CORRECTED_MD_PATH.read_text(encoding="utf-8")
    report = llm_aided_ocr.verify_content_conservation(raw_text, md_text)

    assert report["footnote_count"] == 7
    assert report["footnotes_preserved"] == 7
    assert abs(report["footnote_recall"] - 1.0) < 1e-6
    assert report["missing_footnotes"] == []
    assert report["seam_anomalies"] == []
    assert report["numeric_recall"] >= 0.95
    assert report["deterministic_score"] >= 95


@pytest.mark.asyncio
async def test_b8_assess_output_quality_multiline_and_fallback():
    raw_text = "Page 1:\nRevenue grew 25% to $140,000 in 1975.\n\n* Author footnote on accounting rules."
    proc_text = "Revenue grew 25% to $140,000 in 1975.\n\n> * Author footnote on accounting rules."

    multiline_llm_reply = (
        "SCORE: 96\n"
        "EXPLANATION: Line one of the explanation.\n"
        "Line two with detailed conservation notes.\n"
        "Line three conclusion."
    )
    with patch.object(llm_aided_ocr, "generate_completion", return_value=multiline_llm_reply):
        score, explanation = await llm_aided_ocr.assess_output_quality(raw_text, proc_text)
        assert score == 96
        assert explanation is not None
        assert "Line one of the explanation." in explanation
        assert "Line two with detailed conservation notes." in explanation
        assert "Line three conclusion." in explanation

    # Fallback to deterministic conservation score when LLM returns None
    with patch.object(llm_aided_ocr, "generate_completion", return_value=None):
        score_fb, exp_fb = await llm_aided_ocr.assess_output_quality(raw_text, proc_text)
        assert score_fb is not None and score_fb >= 90
        assert exp_fb is not None and "Deterministic conservation audit:" in exp_fb


@pytest.mark.asyncio
async def test_b9_cli_end_to_end_raw_ocr_and_verify_only(tmp_path, capsys):
    cli = _load_cli_module()

    # 1. Test --verify-only mode on the shipped Warren Buffett files
    verify_args = cli.parse_arguments(["--verify-only", str(RAW_OCR_PATH), str(CORRECTED_MD_PATH)])
    rc = await cli.run_pdf_processor(verify_args)
    assert rc == 0
    try:
        captured = json.loads(capsys.readouterr().out)
    except json.JSONDecodeError as exc:
        pytest.fail(f"CLI --verify-only did not output valid JSON: {exc}")
    assert abs(captured["footnote_recall"] - 1.0) < 1e-6
    assert captured["seam_anomalies"] == []

    # 2. Test --raw-ocr-input mode with mocked LLM completion writing to tmp_path
    sample_raw_file = tmp_path / "sample_doc__raw_ocr_output.txt"
    sample_raw_file.write_text(
        "~2-\n\n# Sample Heading\n\nThis is a sample paragraph with $50,000 and 15% return.\n\n"
        "* Substantive footnote explaining the 15% return assumption in detail.",
        encoding="utf-8",
    )

    async def fake_completion(prompt: str, max_tokens: int = 5000) -> str:
        if "SCORE:" in prompt:
            return "SCORE: 95\nEXPLANATION: Cleanly formatted and conserved."
        if "Current chunk to process:\n" in prompt:
            body = prompt.split("Current chunk to process:\n", 1)[1].split("\n\nCorrected text:", 1)[0]
            return body.strip()
        if "Text to reformat:\n\n" in prompt:
            body = prompt.split("Text to reformat:\n\n", 1)[1].split("\n\nReformatted markdown:", 1)[0]
            return body.strip()
        return prompt

    with patch.object(llm_aided_ocr, "generate_completion", side_effect=fake_completion):
        run_args = cli.parse_arguments(
            [
                "--raw-ocr-input",
                str(sample_raw_file),
                "--output-dir",
                str(tmp_path),
                "--provider",
                "openai",
                "--model",
                "gpt-4o-mini",
            ]
        )
        rc2 = await cli.run_pdf_processor(run_args)
        assert rc2 == 0

    expected_out = tmp_path / "sample_doc_llm_corrected.md"
    assert expected_out.exists()
    out_text = expected_out.read_text(encoding="utf-8")
    assert "~2-" not in out_text
    assert "$50,000" in out_text
    assert "Substantive footnote explaining the 15% return assumption" in out_text


def test_b11_line_strip_deskew_preserves_upright_indented_pages_and_corrects_skew():
    # Page 10 and Page 14 of the Buffett PDF have indented numbered lists/paragraphs that previously
    # caused global cv2.minAreaRect(coords) to falsely tilt upright pages by 6-12 degrees.
    page10 = llm_aided_ocr.convert_pdf_to_images(str(SAMPLE_PDF_PATH), max_pages=1, skip_first_n_pages=9, dpi=300)[0]
    page14 = llm_aided_ocr.convert_pdf_to_images(str(SAMPLE_PDF_PATH), max_pages=1, skip_first_n_pages=13, dpi=300)[0]

    gray10 = np.array(page10.convert("L"))
    gray14 = np.array(page14.convert("L"))

    assert abs(llm_aided_ocr.estimate_skew_angle(gray10)) < 1e-6
    assert abs(llm_aided_ocr.estimate_skew_angle(gray14)) < 1e-6

    ocr10 = llm_aided_ocr.ocr_image(page10, preprocess=True)
    ocr14 = llm_aided_ocr.ocr_image(page14, preprocess=True)
    assert "movement of the tides" in ocr10
    assert "comfort level produced by this option" in ocr14.lower()

    # Now synthetically rotate Page 14 counter-clockwise by +3.5 degrees and verify deskew recovery
    h, w = gray14.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), 3.5, 1.0)
    skewed14 = cv2.warpAffine(gray14, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    est_skew = llm_aided_ocr.estimate_skew_angle(skewed14)
    assert -4.2 <= est_skew <= -2.8
    deskewed14 = llm_aided_ocr.deskew_image(skewed14)
    assert abs(llm_aided_ocr.estimate_skew_angle(deskewed14)) < 1e-6


@pytest.mark.asyncio
async def test_b12_lexical_and_embedding_similarity_and_conservation_metrics():
    assert abs(llm_aided_ocr.compute_lexical_similarity("identical text sample", "identical text sample") - 1.0) < 1e-6
    assert llm_aided_ocr.compute_lexical_similarity("alpha beta gamma", "xyz uvw rst") < 0.15

    raw_text = await asyncio.to_thread(RAW_OCR_PATH.read_text, encoding="utf-8")
    proc_text = await asyncio.to_thread(CORRECTED_MD_PATH.read_text, encoding="utf-8")
    conservation = llm_aided_ocr.verify_content_conservation(raw_text, proc_text)
    assert "lexical_cosine_similarity" in conservation
    assert conservation["lexical_cosine_similarity"] >= 0.94

    # Test offline fallback of compute_embedding_similarity when get_openai_client() is None
    with patch.object(llm_aided_ocr, "get_openai_client", return_value=None):
        sim_offline = await llm_aided_ocr.compute_embedding_similarity(raw_text, proc_text)
        assert sim_offline >= 0.94

    # Test online path of compute_embedding_similarity with mocked OpenAI embeddings response
    class _FakeEmbeddingItem:
        def __init__(self, vec: list[float]):
            self.embedding = vec

    class _FakeEmbeddingResponse:
        def __init__(self):
            self.data = [_FakeEmbeddingItem([1.0, 0.0, 0.0]), _FakeEmbeddingItem([0.8, 0.6, 0.0])]

    class _FakeEmbeddingsAPI:
        async def create(self, model: str, input: list[str]):
            assert model == llm_aided_ocr.OPENAI_EMBEDDING_MODEL
            assert len(input) == 2
            return _FakeEmbeddingResponse()

    class _FakeOpenAIClient:
        def __init__(self):
            self.embeddings = _FakeEmbeddingsAPI()

    with patch.object(llm_aided_ocr, "get_openai_client", return_value=_FakeOpenAIClient()):
        sim_online = await llm_aided_ocr.compute_embedding_similarity("first text", "second text")
        assert abs(sim_online - 0.8) < 1e-6


def test_b13_buffett_sample_markdown_tables_and_heading_hierarchy():
    md_text = CORRECTED_MD_PATH.read_text(encoding="utf-8")
    assert "## Act One - The Awakening" in md_text
    assert "## Act Two - The Great Leap" in md_text
    assert "| Period | S & P 500 | Becker Median Result* |" in md_text
    assert "| Last 3 market cycles, (6/30/62 to 12/31/74) | 5.3% | 4.1% |" in md_text
    assert "| Benchmark / Manager | Annual Compounded Rate of Return (including Dividends) |" in md_text
    assert "| Morgan Guaranty | +3.4% |" in md_text
    assert "## A fixed income (bonds) investment strategy" not in md_text
    assert "## My final option" not in md_text


@pytest.mark.asyncio
async def test_b14_extended_cli_flags_and_json_report_export(tmp_path):
    cli = _load_cli_module()
    report_path = tmp_path / "verify_report.json"

    verify_args = cli.parse_arguments(
        [
            "--verify-only",
            str(RAW_OCR_PATH),
            str(CORRECTED_MD_PATH),
            "--report-json",
            str(report_path),
            "--max-concurrency",
            "6",
            "--chunk-size",
            "6000",
            "--no-preprocess",
            "--no-quality-check",
        ]
    )
    cli.apply_runtime_overrides(verify_args)
    assert llm_aided_ocr.MAX_CONCURRENT_REQUESTS == 6

    rc = await cli.run_pdf_processor(verify_args)
    assert rc == 0
    assert report_path.exists()
    raw_json = await asyncio.to_thread(report_path.read_text, encoding="utf-8")
    try:
        report_data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Report JSON is invalid: {exc}")
    assert abs(report_data["footnote_recall"] - 1.0) < 1e-6
    assert report_data["lexical_cosine_similarity"] >= 0.94
    assert report_data["deterministic_score"] >= 95


@pytest.mark.asyncio
async def test_b16_placeholder_api_keys_return_none_and_local_model_short_circuit():
    with (
        patch.object(llm_aided_ocr, "OPENAI_API_KEY", "your-openai-api-key"),
        patch.object(llm_aided_ocr, "ANTHROPIC_API_KEY", "your-anthropic-api-key"),
    ):
        assert llm_aided_ocr.get_openai_client() is None
        assert llm_aided_ocr.get_anthropic_client() is None

    # Verify download_models() immediately returns existing DEFAULT_LOCAL_MODEL_NAME without network download
    models_dir = Path(llm_aided_ocr.__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    dummy_local = models_dir / "test_b16_existing_local.gguf"
    created = False
    if not dummy_local.exists():
        await asyncio.to_thread(dummy_local.write_bytes, b"GGUF-local")
        created = True
    try:
        with patch.object(llm_aided_ocr, "DEFAULT_LOCAL_MODEL_NAME", "test_b16_existing_local.gguf"):
            names, statuses = await llm_aided_ocr.download_models()
            assert names == ["test_b16_existing_local.gguf"]
            assert statuses and statuses[0]["status"] == "success"
            assert statuses[0]["message"] == "File already exists."
    finally:
        if created and dummy_local.exists():
            await asyncio.to_thread(dummy_local.unlink)


def test_b17_split_document_into_chunks_preserves_paragraph_boundaries():
    oversized_para = " ".join(
        f"Sentence {i} in the long opening paragraph explains pension liability compounding."
        for i in range(1, 35)
    )
    para_two = "Second paragraph has distinct content about actuarial assumptions."
    para_three = "Third paragraph concludes the section."
    doc_text = f"{oversized_para}\n\n{para_two}\n\n{para_three}"

    chunks = llm_aided_ocr.split_document_into_chunks(doc_text, chunk_size=600, model_name="gpt-4o-mini")
    assert len(chunks) >= 3
    # The final chunk must contain para_two and para_three separated by a double newline, not mixed with oversized_para sentences
    assert f"{para_two}\n\n{para_three}" in chunks[-1]
    assert "Sentence 1 " not in chunks[-1]


@pytest.mark.asyncio
async def test_b18_deterministic_reflow_dehyphenate_and_no_llm_cli_mode(tmp_path):
    raw_sample = (
        "~2-\n\n"
        "ACT ONE - THE AWAKENING\n\n"
        "Total pension Silence liabilities compound over long-\n"
        "term horizons when manage-\n"
        "ment makes twenty-\n"
        "five year promises.\n\n"
        "* Author footnote explaining the actuarial assumptions in detail."
    )
    reflowed = llm_aided_ocr.reflow_and_dehyphenate_ocr_text(raw_sample, reformat_as_markdown=True)
    assert "## Act One - The Awakening" in reflowed
    assert "long-term horizons when management makes twenty-five year promises." in reflowed
    assert "* Author footnote explaining the actuarial assumptions in detail." in reflowed

    # End-to-end CLI test with --no-llm and --report-json (guaranteed zero LLM calls)
    cli = _load_cli_module()
    raw_in = tmp_path / "offline_doc__raw_ocr_output.txt"
    report_out = tmp_path / "offline_report.json"
    await asyncio.to_thread(raw_in.write_text, raw_sample, encoding="utf-8")

    with patch.object(llm_aided_ocr, "generate_completion", side_effect=AssertionError("LLM should not be called")):
        args = cli.parse_arguments(
            [
                "--raw-ocr-input",
                str(raw_in),
                "--output-dir",
                str(tmp_path),
                "--no-llm",
                "--report-json",
                str(report_out),
            ]
        )
        rc = await cli.run_pdf_processor(args)
        assert rc == 0

    corrected_out = tmp_path / "offline_doc_llm_corrected.md"
    assert corrected_out.exists()
    out_text = await asyncio.to_thread(corrected_out.read_text, encoding="utf-8")
    assert "~2-" not in out_text
    assert "management makes twenty-five year promises." in out_text
    assert report_out.exists()


def test_b19_ocr_comma_decimal_and_percent_96_conservation_on_buffett_sample():
    # Direct unit test of comma-decimal normalization and 96->% OCR repair recognition
    raw_snippet = "Dow-Jones Industrials +2,8% while bond rates then were 496 instead of 9% on $70,000."
    proc_snippet = "Dow-Jones Industrials +2.8% while bond rates then were 4% instead of 9% on $70,000."
    snippet_report = llm_aided_ocr.verify_content_conservation(raw_snippet, proc_snippet)
    assert snippet_report["missing_numbers"] == []
    assert abs(snippet_report["numeric_recall"] - 1.0) < 1e-6

    # Full flagship Warren Buffett letter conservation check must now achieve 100% numeric recall and 100/100 score
    raw_text = RAW_OCR_PATH.read_text(encoding="utf-8")
    md_text = CORRECTED_MD_PATH.read_text(encoding="utf-8")
    full_report = llm_aided_ocr.verify_content_conservation(raw_text, md_text)
    assert full_report["missing_numbers"] == []
    assert abs(full_report["numeric_recall"] - 1.0) < 1e-6
    assert abs(full_report["footnote_recall"] - 1.0) < 1e-6
    assert full_report["seam_anomalies"] == []
    assert full_report["deterministic_score"] == 100

