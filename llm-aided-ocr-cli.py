import argparse
import asyncio
import json
import sys

import llm_aided_ocr


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a PDF or raw OCR text file with Tesseract OCR, LLM correction, seam stitching, and conservation verification."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="160301289-Warren-Buffett-Katharine-Graham-Letter.pdf",
        help="Path to the input PDF file (default: 160301289-Warren-Buffett-Katharine-Graham-Letter.pdf)",
    )
    parser.add_argument(
        "--raw-ocr-input",
        dest="raw_ocr_input",
        default=None,
        help="Path to an existing raw OCR text file to process instead of running Tesseract on a PDF",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory where output files should be written (defaults to input file directory)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=llm_aided_ocr.PDF_DPI,
        help=f"DPI resolution for PDF-to-image rendering (default: {llm_aided_ocr.PDF_DPI})",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openai", "claude"],
        default=None,
        help="Override the LLM backend provider (local, openai, or claude)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name for the selected provider",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum number of pages to process (0 for all pages)",
    )
    parser.add_argument(
        "--skip-pages",
        type=int,
        default=0,
        help="Number of pages to skip from the beginning",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Do not reformat the output as Markdown",
    )
    parser.add_argument(
        "--keep-headers",
        action="store_true",
        help="Keep running headers and page numbers in the corrected output",
    )
    parser.add_argument(
        "--verify-only",
        nargs=2,
        metavar=("RAW_OCR_FILE", "PROCESSED_FILE"),
        help="Run deterministic content conservation verification comparing RAW_OCR_FILE and PROCESSED_FILE",
    )
    return parser.parse_args(argv)


def apply_runtime_overrides(args: argparse.Namespace) -> None:
    if args.provider:
        provider_lower = args.provider.lower()
        if provider_lower == "local":
            llm_aided_ocr.USE_LOCAL_LLM = True
        elif provider_lower == "openai":
            llm_aided_ocr.USE_LOCAL_LLM = False
            llm_aided_ocr.API_PROVIDER = "OPENAI"
        elif provider_lower == "claude":
            llm_aided_ocr.USE_LOCAL_LLM = False
            llm_aided_ocr.API_PROVIDER = "CLAUDE"

    if args.model:
        if llm_aided_ocr.USE_LOCAL_LLM:
            llm_aided_ocr.DEFAULT_LOCAL_MODEL_NAME = args.model
        elif llm_aided_ocr.API_PROVIDER == "CLAUDE":
            llm_aided_ocr.CLAUDE_MODEL_STRING = args.model
        else:
            llm_aided_ocr.OPENAI_COMPLETION_MODEL = args.model


def run_verification_report(raw_path: str, proc_path: str) -> dict[str, object]:
    with open(raw_path, encoding="utf-8") as rf:
        raw_text = rf.read()
    with open(proc_path, encoding="utf-8") as pf:
        proc_text = pf.read()
    return llm_aided_ocr.verify_content_conservation(raw_text, proc_text)


async def run_pdf_processor(args: argparse.Namespace) -> int:
    if args.verify_only:
        raw_path, proc_path = args.verify_only
        report = await asyncio.to_thread(run_verification_report, raw_path, proc_path)
        print(json.dumps(report, indent=2))
        return 0

    apply_runtime_overrides(args)
    await llm_aided_ocr.main(
        input_pdf_file_path=args.input_file,
        max_test_pages=args.max_pages,
        skip_first_n_pages=args.skip_pages,
        reformat_as_markdown=not args.no_markdown,
        suppress_headers_and_page_numbers=not args.keep_headers,
        output_dir=args.output_dir,
        raw_ocr_input_path=args.raw_ocr_input,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    cli_args = parse_arguments()
    sys.exit(asyncio.run(run_pdf_processor(cli_args)))
