
import streamlit as st
import re
import os
import tempfile
from collections import Counter

from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, PDFPageAggregator # Import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextLineHorizontal, LTChar, LTAnno, LTContainer, LTFigure, LTImage, LTTextBoxHorizontal
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from io import StringIO
import numpy as np # For median calculation

def parse_lt_objs(lt_objs, extracted_data):
    """Recursively parse LT objects and extract text with font info."""
    for lt_obj in lt_objs:
        if isinstance(lt_obj, LTTextLineHorizontal):
            current_line_chars = []
            for char_obj in lt_obj:
                if isinstance(char_obj, LTChar):
                    current_line_chars.append({
                        'text': char_obj.get_text(),
                        'font_size': char_obj.height,
                        'is_bold': "Bold" in char_obj.fontname or "Bd" in char_obj.fontname
                    })
                elif isinstance(char_obj, LTAnno):
                    if char_obj.get_text() == ' ':
                        # Avoid consecutive spaces unless they are significant layout elements
                        if current_line_chars and current_line_chars[-1]['text'] != ' ':
                            current_line_chars.append({'text': ' ', 'font_size': None, 'is_bold': False})
                    elif char_obj.get_text() == '\n':
                        # Explicit newline within a text line, if any
                        if current_line_chars:
                            extracted_data.extend(current_line_chars)
                            extracted_data.append({'text': '\n', 'font_size': None, 'is_bold': False})
                            current_line_chars = [] # Reset for next segment of line
            if current_line_chars:
                extracted_data.extend(current_line_chars)
            # Add a newline at the end of every LTTextLine, unless it's the very last item and already a newline
            if extracted_data and (not extracted_data[-1]['text'] == '\n'):
                extracted_data.append({'text': '\n', 'font_size': None, 'is_bold': False})
        elif isinstance(lt_obj, LTContainer):
            parse_lt_objs(lt_obj._objs, extracted_data) # Recurse for nested objects

# Helper to recursively get all LTTextBoxHorizontal
def _get_all_text_boxes(lt_objs):
    text_boxes = []
    for obj in lt_objs:
        if isinstance(obj, LTTextBoxHorizontal):
            text_boxes.append(obj)
        elif isinstance(obj, LTContainer):
            text_boxes.extend(_get_all_text_boxes(obj._objs)) # Recursive call
    return text_boxes

def collect_text_lines_with_layout_awareness(lt_objs, extracted_data, layout_type):
    """Collect text lines from LT objects, respecting layout type."""
    if layout_type == 'Two Columns':
        text_boxes = _get_all_text_boxes(lt_objs)

        # Sort text boxes by y-coordinate (top to bottom) and then by x-coordinate (left to right)
        # Using negative y0 to sort from top (larger y) to bottom (smaller y)
        text_boxes.sort(key=lambda o: (-o.y0, o.x0))

        for text_box in text_boxes:
            # Process content of each sorted text box
            parse_lt_objs(text_box._objs, extracted_data)
            # Add a newline character at the end of each text box to ensure separation
            # Check if the last item isn't already a newline from parse_lt_objs
            if extracted_data and extracted_data[-1]['text'] != '\n':
                extracted_data.append({'text': '\n', 'font_size': None, 'is_bold': False})

    else: # 'Single Column' or any other default
        parse_lt_objs(lt_objs, extracted_data)

# Function to extract text from PDF with font information
def extract_text_from_pdf(uploaded_file, layout_type):
    if uploaded_file is None:
        return None

    extracted_data = []

    try:
        # Create a temporary file to save the uploaded PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_pdf_path = tmp_file.name

        # Open the PDF file
        with open(tmp_pdf_path, 'rb') as fp:
            rsrcmgr = PDFResourceManager()
            parser = PDFParser(fp) # Need a parser for PDFDocument
            document = PDFDocument(parser) # Initialize document with parser
            laparams = LAParams()

            for page in PDFPage.create_pages(document):
                # Use PDFPageAggregator to get layout objects directly
                device = PDFPageAggregator(rsrcmgr, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                interpreter.process_page(page)
                layout = device.get_result() # Get the aggregated layout object
                device.close()

                # Now 'layout' is an LTPage object, which is an LTContainer
                collect_text_lines_with_layout_awareness(layout._objs, extracted_data, layout_type)

                # Add a page break marker
                extracted_data.append({'text': '\f', 'font_size': None, 'is_bold': False})

        # Clean up the temporary file
        os.remove(tmp_pdf_path)
        return extracted_data
    except Exception as e:
        st.error(f"PDFの読み取り中にエラーが発生しました: {e}")
        # Clean up the temporary file even if an error occurs
        if 'tmp_pdf_path' in locals() and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)
        return None


# Function to clean PDF text artifacts (updated to handle list of dicts)
def clean_pdf_text_artifacts(text_data):
    if not text_data:
        return []

    # First pass: Group characters into logical lines, handle repeated spaces
    processed_lines = []
    current_line_chars = []

    for item in text_data:
        text = item['text']

        if text == '\n': # End of a physical line detected by pdfminer
            if current_line_chars:
                # Combine characters in the current_line_chars into a single line object
                line_text = ''.join([char_item['text'] for char_item in current_line_chars]).strip()
                if line_text: # Only add if the line has actual content
                    # Take font info from the first actual character in the line
                    first_char_info = next((c for c in current_line_chars if c['text'].strip()), {'font_size': None, 'is_bold': False})
                    processed_lines.append({
                        'text': line_text,
                        'font_size': first_char_info['font_size'],
                        'is_bold': first_char_info['is_bold']
                    })
                current_line_chars = [] # Reset for next line
            # Only append newline if it's not directly followed by another newline or page break in processed_lines
            if not (processed_lines and (processed_lines[-1]['text'] == '\n' or processed_lines[-1]['text'] == '\f')):
                processed_lines.append({'text': '\n', 'font_size': None, 'is_bold': False}) # Add the newline marker
        elif text == '\f': # Page break marker
            if current_line_chars:
                line_text = ''.join([char_item['text'] for char_item in current_line_chars]).strip()
                if line_text:
                    first_char_info = next((c for c in current_line_chars if c['text'].strip()), {'font_size': None, 'is_bold': False})
                    processed_lines.append({
                        'text': line_text,
                        'font_size': first_char_info['font_size'],
                        'is_bold': first_char_info['is_bold']
                    })
                current_line_chars = []
            if not (processed_lines and processed_lines[-1]['text'] == '\f'):
                processed_lines.append({'text': '\f', 'font_size': None, 'is_bold': False}) # Add page break marker
        else: # Regular characters
            current_line_chars.append(item)

    # Add any remaining characters as a final line
    if current_line_chars:
        line_text = ''.join([char_item['text'] for char_item in current_line_chars]).strip()
        if line_text:
            first_char_info = next((c for c in current_line_chars if c['text'].strip()), {'font_size': None, 'is_bold': False})
            processed_lines.append({
                'text': line_text,
                'font_size': first_char_info['font_size'],
                'is_bold': first_char_info['is_bold']
            })

    # Second pass: Handle vertical duplication of lines and multiple newlines
    final_cleaned_structured_data = []
    i = 0
    while i < len(processed_lines):
        current_item = processed_lines[i]

        if current_item['text'] == '\n':
            # Consolidate multiple newlines to max two (for paragraph breaks)
            if not (final_cleaned_structured_data and final_cleaned_structured_data[-1]['text'] == '\n'):
                final_cleaned_structured_data.append(current_item)
            elif final_cleaned_structured_data and final_cleaned_structured_data[-1]['text'] == '\n' and \
                 len([item for item in final_cleaned_structured_data[::-1] if item['text'] == '\n']) < 2:
                final_cleaned_structured_data.append(current_item)
            i += 1
            continue
        elif current_item['text'] == '\f': # Always keep single page break
            final_cleaned_structured_data.append(current_item)
            i += 1
            continue

        # Check for vertical duplication only with text content
        # Ensure we don't go out of bounds and check only text lines
        if i + 1 < len(processed_lines) and processed_lines[i+1]['text'] not in ['\n', '\f']:
            next_item = processed_lines[i+1]
            if current_item['text'] == next_item['text'] and \
               current_item['font_size'] == next_item['font_size'] and \
               current_item['is_bold'] == next_item['is_bold']: # Compare font info too
                # If they are identical lines, add only one and skip the next
                final_cleaned_structured_data.append(current_item)
                i += 2 # Skip current and next identical line
            else:
                final_cleaned_structured_data.append(current_item)
                i += 1
        else: # No next line or next is a newline/page break
            final_cleaned_structured_data.append(current_item)
            i += 1

    return final_cleaned_structured_data

def detect_heading_patterns_llm_like(cleaned_text_data):
    base_patterns_str = {
        r'^[A-Z]\.\s', # E.g., 'A. Section Title'
        r'^\d+\.\s', # E.g., '1. Section Title', '1.1. Subtitle'
        r'^■', # Bullet point style heading
        r'^・', # Another bullet point style
        r'^【.+】', # Text enclosed in brackets
        r'^\( ?[0-9a-zA-Zア-ンー]+\)', # (1), (a), (イ)
        r'^\[ ?[0-9a-zA-Zア-ンー]+\]', # [1], [a], [イ]
        r'^[A-Z]章$', # E.g., 'A章'
        r'^第[一二三四五六七八九十百千万億]+[章条]$', # E.g., '第一章', '第一条' - UPDATED
        r'^[0-9]+[．\.．]\s',
        r'^[0-9]+[ ]?[項目編]', # 1項目, 2編
        r'^[0-9A-Za-zア-ンー]{1,3}節',
        r'^[0-9]+\.[0-9]+\s',
        r'^□', # NEW: Covers '□舞台', '□あらすじ' etc.
        r'^▼', # NEW: Covers '▼はじめに'
        r'^▷', # NEW: Covers '▷見た映像をそのままデータに記録する記録機能'
        r'^❖', # NEW: Covers '❖共通 HO' etc.
        r'^目次$', # NEW: Covers '目次'
        r'^●$', # NEW: Covers '目次' (common bullet)
        r'^◇$', # NEW: Covers '目次' (common bullet)
        r'^★$'  # NEW: Covers '目次' (common bullet)
    }

    detected_new_patterns_str = set()
    all_font_sizes = []

    # Compile base patterns for internal use
    compiled_base_patterns = [re.compile(p) for p in base_patterns_str]

    # Collect all font sizes to determine a baseline/median for general text
    for item in cleaned_text_data:
        if item['text'].strip() and item['font_size'] is not None:
            all_font_sizes.append(item['font_size'])

    median_font_size = np.median(all_font_sizes) if all_font_sizes else 0
    # Heuristic: a general heading is likely to be larger than the average text, e.g., 1.1x median
    general_font_size_threshold = median_font_size * 1.1 if median_font_size else 0

    # --- New logic: Analyze characteristics of lines matching base patterns ---
    strong_heading_font_sizes_from_base = []
    strong_heading_bold_status_from_base = []

    # Convert cleaned_text_data to a simpler list of lines for easier iteration with index
    lines_info = []
    for item in cleaned_text_data:
        if item['text'] not in ['\n', '\f']:
            lines_info.append(item)
        else: # Mark line breaks for context
            lines_info.append({'text': item['text']}) # Keep only 'text' for non-content items

    for i, line_item in enumerate(lines_info):
        stripped_line = line_item.get('text', '').strip()

        if not stripped_line or line_item['text'] in ['\n', '\f']:
            continue

        # Check if this line matches any of the base patterns
        is_base_pattern_match = False
        for pattern in compiled_base_patterns:
            if pattern.search(stripped_line):
                is_base_pattern_match = True
                break

        if is_base_pattern_match:
            if line_item.get('font_size') is not None:
                strong_heading_font_sizes_from_base.append(line_item['font_size'])
            strong_heading_bold_status_from_base.append(line_item.get('is_bold', False))

    # Calculate specific thresholds based on observed strong headings
    specific_heading_font_size_threshold = general_font_size_threshold # Default to general if no base patterns found
    if strong_heading_font_sizes_from_base:
        # Take the median + a small margin for more flexibility
        specific_median_heading_font_size = np.median(strong_heading_font_sizes_from_base)
        specific_heading_font_size_threshold = specific_median_heading_font_size * 1.05 # Slightly lower multiplier or adapt

    # Determine if bold is a strong indicator among base patterns
    prefer_bold_for_new_headings = False
    if strong_heading_bold_status_from_base:
        bold_count = sum(1 for b in strong_heading_bold_status_from_base if b)
        if bold_count / len(strong_heading_bold_status_from_base) > 0.6: # More than 60% of base headings are bold
            prefer_bold_for_new_headings = True
    # --- End new logic ---


    for i, line_item in enumerate(lines_info):
        stripped_line = line_item.get('text', '').strip()

        if not stripped_line or line_item['text'] in ['\n', '\f']:
            continue

        # Skip obvious page numbers or single digit lines (heuristic)
        if re.match(r'^(P\.\d+|\d+)$', stripped_line) or (stripped_line.isdigit() and len(stripped_line) < 3):
            continue

        # Check for surrounding empty lines
        preceded_by_newline_count = 0
        for j in range(i - 1, -1, -1):
            if lines_info[j].get('text') == '\n':
                preceded_by_newline_count += 1
            elif lines_info[j].get('text') == '\f':
                preceded_by_newline_count = 2
                break
            elif lines_info[j].get('text', '').strip():
                break

        followed_by_newline_count = 0
        for j in range(i + 1, len(lines_info)):
            if lines_info[j].get('text') == '\n':
                followed_by_newline_count += 1
            elif lines_info[j].get('text') == '\f':
                followed_by_newline_count = 2
                break
            elif lines_info[j].get('text', '').strip():
                break

        is_surrounded_by_empty = (preceded_by_newline_count >= 1 and followed_by_newline_count >= 1)

        is_bold_feature = line_item.get('is_bold', False)
        font_size_feature = line_item.get('font_size', 0)

        # Combination of heuristics for potential new headings:
        # A line is considered a potential heading if it meets certain criteria:
        if (2 <= len(stripped_line) <= 35) and is_surrounded_by_empty:
            is_strong_candidate = False
            if font_size_feature > specific_heading_font_size_threshold:
                if prefer_bold_for_new_headings:
                    if is_bold_feature:
                        is_strong_candidate = True
                else: # No strong preference for bold, just large font is enough
                    is_strong_candidate = True
            # If not meeting specific font size threshold, check general font size threshold with bold
            elif is_bold_feature and font_size_feature > general_font_size_threshold:
                is_strong_candidate = True
            # If we strongly prefer bold, and the line is bold (even if not significantly larger font)
            elif is_bold_feature and prefer_bold_for_new_headings:
                 is_strong_candidate = True


            if is_strong_candidate:
                is_already_covered_by_generic = False
                for bp_str in [r'^[A-Z]\.\s', r'^■', r'D-\d']:
                    if re.search(bp_str, stripped_line):
                        is_already_covered_by_generic = True
                        break

                if not is_already_covered_by_generic:
                    is_exact_static_match = False
                    for bp_str in base_patterns_str:
                        # Simple heuristic to check if it's a fixed string pattern without special regex characters
                        if not any(char in bp_str for char in ".*+?{}[()]|^$"):
                            if stripped_line == bp_str:
                                is_exact_static_match = True
                                break
                    if not is_exact_static_match:
                        detected_new_patterns_str.add(re.escape(stripped_line))

    all_patterns_str = list(base_patterns_str.union(detected_new_patterns_str))
    return all_patterns_str

# Function to format extracted text (updated to accept dynamic patterns as raw strings)
def format_extracted_text(text_data, heading_pattern_strings):
    if not text_data:
        return ""

    # Compile the heading patterns internally
    compiled_heading_patterns = [re.compile(p) for p in heading_pattern_strings]

    formatted_parts = []
    current_text_block_content = []

    # Regex to find patterns like STR12, CON10 etc. and insert a space
    # Using \b for word boundaries to avoid matching partial words
    stat_insert_space_pattern = re.compile(r'\b([A-Z]{3})(\d+)\b')
    # Regex to identify a line consisting solely of such stat patterns (possibly with spaces)
    stat_line_identifier_pattern = re.compile(r'^(?:[A-Z]{3}\d+\s*)+$')


    for item in text_data:
        line_text = item['text'].strip()

        if not line_text and item['text'] not in ['\n', '\f']: # Ignore completely empty strings that are not structural breaks
            continue

        # First, check if current line is a heading
        is_heading = False
        for pattern in compiled_heading_patterns:
            if pattern.search(line_text):
                is_heading = True
                break

        # If it's a structural break or a heading, flush the current text block
        if item['text'] == '\n' or item['text'] == '\f' or is_heading:
            if current_text_block_content:
                # Join accumulated non-heading lines with a single space
                paragraph_text = ' '.join(current_text_block_content)
                # Apply the stat formatting
                paragraph_text = stat_insert_space_pattern.sub(r'\1 \2', paragraph_text)
                # Consolidate multiple spaces to a single space
                paragraph_text = re.sub(r'\s{2,}', ' ', paragraph_text)
                formatted_parts.append(paragraph_text.strip())
                current_text_block_content = [] # Reset for new text block

            # Handle newlines and page breaks
            if item['text'] == '\n':
                # Consolidate multiple newlines to max two for paragraph breaks
                if not (formatted_parts and formatted_parts[-1] == '\n'):
                    formatted_parts.append('\n')
                elif formatted_parts and formatted_parts[-1] == '\n' and \
                     len([p for p in formatted_parts[::-1] if p == '\n']) < 2: # Limit to two consecutive newlines
                    formatted_parts.append('\n')
            elif item['text'] == '\f':
                if not (formatted_parts and formatted_parts[-1] == '\n'):
                    formatted_parts.append('\n')
                formatted_parts.append('\f\n')

            if is_heading:
                # Add two newlines around the heading
                if formatted_parts and not formatted_parts[-1].endswith('\n\n') and formatted_parts[-1] != '\n':
                    formatted_parts.append('\n\n')
                elif formatted_parts and formatted_parts[-1] == '\n':
                    formatted_parts[-1] = '\n\n'

                formatted_parts.append(line_text)
                formatted_parts.append('\n\n')
            continue # Continue to next item after handling structural break/heading


        # If it's not a structural break or a heading, it's content.
        # Apply the stat-space insertion immediately to the line_text itself
        processed_line = stat_insert_space_pattern.sub(r'\1 \2', line_text)
        processed_line = re.sub(r'\s{2,}', ' ', processed_line).strip() # Consolidate spaces within the line

        # Check if this processed line looks like a stat-only line, to give it its own paragraph
        # We check the original line_text with spaces removed for a full match against stat_line_identifier_pattern
        if stat_line_identifier_pattern.fullmatch(line_text.replace(' ', '')) and processed_line: # Check original text with spaces removed
            if current_text_block_content: # Flush any preceding accumulated text
                paragraph_text = ' '.join(current_text_block_content)
                paragraph_text = stat_insert_space_pattern.sub(r'\1 \2', paragraph_text)
                paragraph_text = re.sub(r'\s{2,}', ' ', paragraph_text)
                formatted_parts.append(paragraph_text.strip())
                current_text_block_content = []

            formatted_parts.append(processed_line)
            formatted_parts.append('\n') # Add a newline after the stat line as requested
        else:
            # For non-heading, non-stat-only lines, accumulate their content
            # If current_text_block_content is empty, and last formatted part is a newline, remove it to concatenate closely
            if not current_text_block_content and formatted_parts and formatted_parts[-1] == '\n':
                formatted_parts.pop() # Remove the single newline if content follows it to join paragraphs
            current_text_block_content.append(processed_line)


    # Add any remaining text block at the end of the document
    if current_text_block_content:
        paragraph_text = ' '.join(current_text_block_content)
        paragraph_text = stat_insert_space_pattern.sub(r'\1 \2', paragraph_text)
        paragraph_text = re.sub(r'\s{2,}', ' ', paragraph_text)
        formatted_parts.append(paragraph_text.strip())

    final_result = "".join(formatted_parts).strip()
    # Consolidate multiple newlines to exactly two for section breaks, and single space within text blocks
    final_result = re.sub(r'(\n\n)+', '\n\n', final_result) # Consolidate multiple newlines to exactly two
    final_result = re.sub(r'\n\f\n', '\f\n', final_result) # Clean up newlines around form feed

    return final_result

# Main Streamlit application logic
def main():
    st.title("PDF Text Extractor and Formatter")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Add radio button for layout type selection
    layout_type = st.radio(
        "Select PDF Layout Type:",
        ("Single Column", "Two Columns"),
        index=0,  # Default to 'Single Column'
        help="Choose 'Single Column' for standard PDFs or 'Two Columns' for PDFs with two-column text layout."
    )

    if uploaded_file is not None:
        st.write("Processing PDF...")

        # Extract text with font info, passing the selected layout type
        extracted_data_with_fonts = extract_text_from_pdf(uploaded_file, layout_type)

        if extracted_data_with_fonts:
            st.subheader("Extracted Data with Font Information (Raw)")
            with st.expander("Show raw extracted data"):
                st.json(extracted_data_with_fonts[:200]) # Show first 200 items

            cleaned_text_data = clean_pdf_text_artifacts(extracted_data_with_fonts)
            st.subheader("Cleaned Structured Text (Lines & Duplicates Processed)")
            with st.expander("Show cleaned structured text"):
                st.json(cleaned_text_data[:200]) # Show first 200 items

            # Get potential heading pattern strings
            all_potential_heading_patterns = detect_heading_patterns_llm_like(cleaned_text_data)

            # Define static patterns that should be pre-selected
            # This set should match the base_patterns_str from detect_heading_patterns_llm_like
            pre_selected_static_patterns = [
                r'^[A-Z]\.\s', # E.g., 'A. Section Title'
                r'^\d+\.\s', # E.g., '1. Section Title', '1.1. Subtitle'
                r'^■', # Bullet point style heading
                r'^・', # Another bullet point style
                r'^【.+】', # Text enclosed in brackets
                r'^\( ?[0-9a-zA-Zア-ンー]+\)', # (1), (a), (イ)
                r'^\[ ?[0-9a-zA-Zア-ンー]+\]', # [1], [a], [イ]
                r'^[A-Z]章$', # E.g., 'A章'
                r'^第[一二三四五六七八九十百千万億]+[章条]$', # E.g., '第一章', '第一条' - UPDATED
                r'^[0-9]+[．\.．]\s',
                r'^[0-9]+[ ]?[項目編]', # 1項目, 2編
                r'^[0-9A-Za-zア-ンー]{1,3}節',
                r'^[0-9]+\.[0-9]+\s',
                r'^□', # NEW: Covers '□舞台', '□あらすじ' etc.
                r'^▼', # NEW: Covers '▼はじめに'
                r'^▷', # NEW: Covers '▷見た映像をそのままデータに記録する記録機能'
                r'^❖', # NEW: Covers '❖共通 HO' etc.
                r'^目次$', # NEW: Covers '目次'
                r'^●$', # Covers '目次' (common bullet)
                r'^◇$', # NEW: Covers '目次' (common bullet)
                r'^★$'  # NEW: Covers '目次' (common bullet)
            ]
            # Ensure pre-selected patterns are in the available options
            default_selected_patterns = [p for p in pre_selected_static_patterns if p in all_potential_heading_patterns]

            st.subheader("Select Heading Patterns")
            selected_patterns = st.multiselect(
                "Choose patterns to treat as headings:",
                options=all_potential_heading_patterns,
                default=default_selected_patterns,
                help="Select the regular expression patterns that should define headings. Text matching these patterns will be separated by two newlines, and other text will be concatenated without spaces."
            )

            # Format text using the selected heading patterns
            formatted_text = format_extracted_text(cleaned_text_data, selected_patterns)

            st.subheader("Formatted Text")
            with st.expander("Show formatted text"):
                st.text(formatted_text)

            # Add download button for formatted text
            st.download_button(
                label="Download Formatted Text",
                data=formatted_text,
                file_name="formatted_text.txt",
                mime="text/plain"
            )
        else:
            st.error("Failed to extract text from the PDF.")

if __name__ == "__main__":
    main()