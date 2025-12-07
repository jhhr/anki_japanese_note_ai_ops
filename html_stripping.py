import re
from html.parser import HTMLParser
from html import unescape


class HTMLStripper(HTMLParser):
    """Custom HTML parser to strip tags while preserving text"""

    def __init__(self):
        super().__init__()
        self.strict = False
        self.convert_charrefs = True
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return "".join(self.text)


def strip_html(html_content):
    """
    Strip HTML tags and clean content for LLM processing

    Args:
        html_content: HTML string from dictionary

    Returns:
        Cleaned plain text string
    """
    if not html_content:
        return ""

    # Use custom HTML stripper
    stripper = HTMLStripper()
    stripper.feed(html_content)
    text = stripper.get_text()

    # Unescape HTML entities
    text = unescape(text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)  # Multiple spaces to single space
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Multiple newlines to double newline
    text = text.strip()

    return text


def strip_html_advanced(html_content, preserve_structure=False):
    """
    Advanced HTML stripping with optional structure preservation

    Args:
        html_content: HTML string from dictionary
        preserve_structure: If True, preserve line breaks and basic formatting

    Returns:
        Cleaned plain text string
    """
    if not html_content:
        return ""

    text = html_content

    if preserve_structure:
        # Convert some HTML elements to text equivalents
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<li>", "• ", text, flags=re.IGNORECASE)
        text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)

    # Remove script and style elements completely
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Unescape HTML entities
    text = unescape(text)

    # Clean up whitespace
    if preserve_structure:
        text = re.sub(r" +", " ", text)  # Multiple spaces to single
        text = re.sub(r"\n\n+", "\n\n", text)  # Limit to double newlines
    else:
        text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text
