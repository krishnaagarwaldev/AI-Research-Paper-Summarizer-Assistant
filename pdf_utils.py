# pdf_utils.py
# Handles extracting readable text from an uploaded PDF file.
# Uses the `pypdf` library (pip install pypdf).

import pypdf
import io


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract and return all text from a Streamlit UploadedFile (PDF).

    Args:
        uploaded_file: A Streamlit UploadedFile object (BytesIO-compatible).

    Returns:
        A single string containing all extracted text, pages separated by newlines.

    Raises:
        ValueError: If the PDF has no extractable text (e.g. scanned image PDF).
        RuntimeError: If pypdf fails to read the file.
    """
    try:
        # Read the uploaded file bytes into a BytesIO buffer
        file_bytes = uploaded_file.read()
        pdf_buffer = io.BytesIO(file_bytes)

        reader = pypdf.PdfReader(pdf_buffer)

        if len(reader.pages) == 0:
            raise ValueError("The uploaded PDF has no pages.")

        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text.strip())

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            raise ValueError(
                "No readable text could be extracted from this PDF. "
                "It may be a scanned image PDF. Please try copy-pasting the text manually."
            )

        return full_text

    except ValueError:
        # Re-raise our descriptive errors as-is
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}") from e


def truncate_text(text: str, max_words: int = 3000) -> str:
    """
    Truncate text to a maximum number of words to avoid overwhelming the API.

    Args:
        text:      The input text string.
        max_words: Maximum number of words to keep (default 3000).

    Returns:
        The (possibly truncated) text string.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    return truncated + "\n\n[Text truncated to first 3000 words for API compatibility.]"
