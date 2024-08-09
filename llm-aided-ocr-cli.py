import argparse
import asyncio
from llm_aided_ocr import main as process_pdf


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a PDF file with OCR and LLM correction.")
    parser.add_argument("input_file", help="Path to the input PDF file")
    parser.add_argument("--max-pages", type=int, default=0, help="Maximum number of pages to process (0 for all pages)")
    parser.add_argument("--skip-pages", type=int, default=0, help="Number of pages to skip from the beginning")
    parser.add_argument("--threshold", type=float, default=0.40, help="Starting hallucination similarity threshold")
    parser.add_argument("--check-english", action="store_true", help="Check if the extracted text is valid English")
    parser.add_argument("--no-markdown", action="store_true", help="Don't reformat the output as markdown")
    parser.add_argument("--db-path", default="./sentence_embeddings.sqlite", help="Path to the sentence embeddings database")
    parser.add_argument("--test-filtering", action="store_true", help="Test hallucination filtering on existing output")
    return parser.parse_args()

async def run_pdf_processor(args):
    await process_pdf(
        input_pdf_file_path=args.input_file,
        max_test_pages=args.max_pages,
        skip_first_n_pages=args.skip_pages,
        starting_hallucination_similarity_threshold=args.threshold,
        check_if_valid_english=args.check_english,
        reformat_as_markdown=not args.no_markdown,
        sentence_embeddings_db_path=args.db_path,
        test_filtering_hallucinations=args.test_filtering
    )
    
if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(run_pdf_processor(args))