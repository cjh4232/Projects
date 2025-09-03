"""
Text linearizer for PDF invoice extraction.

This module transforms PDF text to be sequential and easier to parse,
fixing issues with side-by-side layouts and table structures that 
cause PDF text extraction to fail.
"""

import re
from typing import List, Dict, Tuple, Optional


def linearize_text(text: str) -> str:
    """
    Master preprocessing function to make PDF text look like OCR output.
    
    This fixes the main issues with PDF text extraction:
    1. Side-by-side columns (FROM: ... TO: layouts)
    2. Formatting artifacts (>>>, special chars)
    3. Split table rows that should be combined
    
    Args:
        text: Raw text from PDF extraction
        
    Returns:
        Linearized text that's easier to parse
    """
    # Step 1: Fix side-by-side column layouts
    text = linearize_side_by_side_columns(text)
    
    # Step 2: Clean formatting artifacts
    text = clean_formatting_artifacts(text)
    
    # Step 3: Reconstruct split table rows
    text = reconstruct_table_rows(text)
    
    return text


def linearize_side_by_side_columns(text: str) -> str:
    """
    Detect and fix side-by-side layouts like:
    FROM:          TO:
    Company A      Company B
    
    Convert to:
    FROM:
    Company A
    TO:  
    Company B
    
    This fixes the main issue where vendor/customer info gets mixed up.
    """
    lines = text.split('\n')
    result_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for side-by-side patterns
        side_by_side_match = _detect_side_by_side_layout(line)
        
        if side_by_side_match:
            # Process this side-by-side section
            linearized_section = _linearize_section(lines, i, side_by_side_match)
            result_lines.extend(linearized_section['lines'])
            i = linearized_section['next_index']
        else:
            # Keep line as-is
            result_lines.append(line)
            i += 1
    
    return '\n'.join(result_lines)


def _detect_side_by_side_layout(line: str) -> Optional[Dict]:
    """
    Detect if a line contains side-by-side column headers.
    
    Common patterns:
    - "FROM:          TO:"
    - "VENDOR:       CLIENT:"
    - "Bill From:    Bill To:"
    - Headers separated by 10+ spaces
    """
    # Pattern 1: FROM/TO layout
    from_to_pattern = r'^(FROM\s*:?)\s{10,}(TO\s*:?)$'
    match = re.search(from_to_pattern, line, re.IGNORECASE)
    if match:
        return {
            'type': 'from_to',
            'left_header': match.group(1).strip(),
            'right_header': match.group(2).strip(),
            'left_pos': match.start(1),
            'right_pos': match.start(2)
        }
    
    # Pattern 2: VENDOR/CLIENT layout
    vendor_client_pattern = r'^(VENDOR\s*:?|BILLED?\s*BY\s*:?)\s{8,}(CLIENT\s*:?|CUSTOMER\s*:?)$'
    match = re.search(vendor_client_pattern, line, re.IGNORECASE)
    if match:
        return {
            'type': 'vendor_client',
            'left_header': match.group(1).strip(),
            'right_header': match.group(2).strip(),
            'left_pos': match.start(1),
            'right_pos': match.start(2)
        }
    
    # Pattern 3: Bill From/To layout
    bill_pattern = r'^(BILL\s*FROM\s*:?)\s{8,}(BILL\s*TO\s*:?)$'
    match = re.search(bill_pattern, line, re.IGNORECASE)
    if match:
        return {
            'type': 'bill_from_to',
            'left_header': match.group(1).strip(),
            'right_header': match.group(2).strip(),
            'left_pos': match.start(1),
            'right_pos': match.start(2)
        }
    
    # Pattern 4: Generic two-column detection (any text with 10+ spaces)
    generic_pattern = r'^([A-Za-z][A-Za-z\s]{2,}:?)\s{10,}([A-Za-z][A-Za-z\s]{2,}:?)$'
    match = re.search(generic_pattern, line)
    if match:
        # Only if both parts look like headers (have colons or are short)
        left = match.group(1).strip()
        right = match.group(2).strip()
        if (':' in left or len(left) < 15) and (':' in right or len(right) < 15):
            return {
                'type': 'generic',
                'left_header': left,
                'right_header': right,
                'left_pos': match.start(1),
                'right_pos': match.start(2)
            }
    
    return None


def _linearize_section(lines: List[str], start_idx: int, layout_info: Dict) -> Dict:
    """
    Linearize a side-by-side section starting at start_idx.
    
    Returns:
        Dict with 'lines' (linearized lines) and 'next_index' (where to continue)
    """
    result_lines = []
    left_column = []
    right_column = []
    
    # Column positions from the header line
    left_pos = layout_info['left_pos']
    right_pos = layout_info['right_pos']
    
    # Add the headers as separate lines
    result_lines.append(layout_info['left_header'])
    result_lines.append(layout_info['right_header'])
    
    # Process subsequent lines to extract column data
    i = start_idx + 1
    while i < len(lines):
        line = lines[i]
        
        # Stop if we hit an empty line or a new section
        if not line.strip():
            break
            
        # Stop if this looks like a new header or table start
        if _is_new_section_start(line):
            break
            
        # Extract content for each column based on position
        column_data = _extract_column_content(line, left_pos, right_pos)
        
        if column_data['left']:
            left_column.append(column_data['left'])
        if column_data['right']:
            right_column.append(column_data['right'])
            
        i += 1
    
    # Add left column data
    result_lines.extend(left_column)
    
    # Add right column data  
    result_lines.extend(right_column)
    
    return {
        'lines': result_lines,
        'next_index': i
    }


def _extract_column_content(line: str, left_pos: int, right_pos: int) -> Dict[str, str]:
    """
    Extract content from left and right columns based on position.
    
    Uses the header positions to determine where each column should be.
    """
    line_length = len(line)
    
    # Calculate column boundaries with some tolerance
    left_end = min(right_pos - 5, line_length)  # Leave 5 char buffer
    right_start = min(right_pos - 2, line_length)  # Start 2 chars before right header pos
    
    left_content = ""
    right_content = ""
    
    # Extract left column (from start to left_end)
    if left_end > 0:
        left_content = line[:left_end].strip()
    
    # Extract right column (from right_start to end)
    if right_start < line_length:
        right_content = line[right_start:].strip()
    
    return {
        'left': left_content if left_content else "",
        'right': right_content if right_content else ""
    }


def _is_new_section_start(line: str) -> bool:
    """
    Check if this line indicates the start of a new section.
    
    This helps determine when to stop processing a side-by-side section.
    """
    line_lower = line.lower().strip()
    
    # Common section starters
    section_markers = [
        'invoice', 'description', 'qty', 'quantity', 'price', 'amount', 'total',
        'subtotal', 'tax', 'payment', 'notes', 'terms'
    ]
    
    # If the line starts with any section marker
    for marker in section_markers:
        if line_lower.startswith(marker):
            return True
            
    # If it's a table header (has multiple column-like words)
    header_words = ['description', 'qty', 'price', 'total', 'amount']
    word_count = sum(1 for word in header_words if word in line_lower)
    if word_count >= 2:
        return True
        
    return False


def clean_formatting_artifacts(text: str) -> str:
    """
    Remove formatting artifacts that interfere with parsing.
    
    Common artifacts from PDF extraction:
    - Lines starting with ">>>"
    - Excessive whitespace
    - Special characters that don't belong
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove lines that are just formatting artifacts
        if line.strip().startswith('>>>'):
            continue
            
        if line.strip() in ['---', '===', '___']:
            continue
            
        # Clean up excessive whitespace but preserve structure
        cleaned_line = re.sub(r'\s{2,}', ' ', line).strip()
        
        # Only keep non-empty lines
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


def reconstruct_table_rows(text: str) -> str:
    """
    Reconstruct table rows that were split across multiple lines.
    
    This fixes cases where:
    Description
    Qty
    Unit Price  
    Total
    
    Should become:
    Description | Qty | Unit Price | Total
    """
    lines = text.split('\n')
    result_lines = []
    
    # Find table regions
    table_regions = _find_table_regions(lines)
    
    if not table_regions:
        # No tables found, return as-is
        return text
    
    current_line_idx = 0
    
    for region in table_regions:
        # Add lines before this table region
        while current_line_idx < region['start']:
            result_lines.append(lines[current_line_idx])
            current_line_idx += 1
        
        # Process the table region
        reconstructed_table = _reconstruct_table_region(
            lines[region['start']:region['end']]
        )
        result_lines.extend(reconstructed_table)
        
        current_line_idx = region['end']
    
    # Add remaining lines
    while current_line_idx < len(lines):
        result_lines.append(lines[current_line_idx])
        current_line_idx += 1
    
    return '\n'.join(result_lines)


def _find_table_regions(lines: List[str]) -> List[Dict]:
    """
    Find regions that look like tables (headers followed by data rows).
    """
    regions = []
    
    for i, line in enumerate(lines):
        if _is_table_header(line):
            # Found a potential table header, look for the end
            end_idx = _find_table_end(lines, i + 1)
            if end_idx > i + 1:  # Must have at least one data row
                regions.append({
                    'start': i,
                    'end': end_idx,
                    'type': 'table'
                })
    
    return regions


def _is_table_header(line: str) -> bool:
    """
    Check if a line looks like a table header.
    """
    line_lower = line.lower()
    
    # Look for common table headers
    header_indicators = ['description', 'qty', 'quantity', 'price', 'total', 'amount']
    
    # Count how many header words are in this line
    header_count = sum(1 for word in header_indicators if word in line_lower)
    
    return header_count >= 2


def _find_table_end(lines: List[str], start_idx: int) -> int:
    """
    Find where a table ends (when we hit totals or empty space).
    """
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        
        if not line:
            # Empty line might be end of table
            return i
            
        line_lower = line.lower()
        
        # Stop at summary lines
        if any(word in line_lower for word in ['subtotal', 'total:', 'tax', 'payment']):
            return i
            
        # If we've gone too far without finding data, stop
        if i - start_idx > 20:  # Max 20 lines for a table
            return i
    
    return len(lines)


def _reconstruct_table_region(table_lines: List[str]) -> List[str]:
    """
    Reconstruct a table region by combining split rows.
    
    Strategy:
    1. Identify if headers are split across lines
    2. If so, reconstruct header and data rows
    3. Otherwise, return as-is
    """
    if not table_lines:
        return []
    
    # Check if we have split headers
    header_analysis = _analyze_table_headers(table_lines)
    
    if header_analysis['is_split']:
        return _reconstruct_split_table(table_lines, header_analysis)
    else:
        # Table is already in good format
        return table_lines


def _analyze_table_headers(table_lines: List[str]) -> Dict:
    """
    Analyze if table headers are split across multiple lines.
    """
    if not table_lines:
        return {'is_split': False}
    
    # Look at first few lines to see if they're individual header words
    header_words = []
    potential_header_lines = 0
    
    for i, line in enumerate(table_lines[:5]):  # Check first 5 lines
        line_stripped = line.strip().lower()
        
        if line_stripped in ['description', 'qty', 'quantity', 'unit price', 'price', 'total', 'amount']:
            header_words.append(line_stripped)
            potential_header_lines = i + 1
        else:
            break
    
    # If we found 3+ header words in separate lines, it's likely split
    is_split = len(header_words) >= 3 and potential_header_lines >= 3
    
    return {
        'is_split': is_split,
        'header_words': header_words,
        'header_line_count': potential_header_lines
    }


def _reconstruct_split_table(table_lines: List[str], header_analysis: Dict) -> List[str]:
    """
    Reconstruct a table where headers and data are split across lines.
    """
    reconstructed = []
    
    # Reconstruct header
    header_parts = []
    for i in range(header_analysis['header_line_count']):
        if i < len(table_lines):
            header_parts.append(table_lines[i].strip())
    
    # Create combined header
    reconstructed.append(' | '.join(header_parts))
    
    # Process data rows
    data_start = header_analysis['header_line_count']
    columns_per_row = len(header_parts)
    
    i = data_start
    while i < len(table_lines):
        # Skip empty lines
        if not table_lines[i].strip():
            i += 1
            continue
            
        # Stop at summary lines
        line_lower = table_lines[i].lower()
        if any(word in line_lower for word in ['subtotal', 'total:', 'tax', 'payment']):
            # Add remaining lines as-is (they're summary, not table data)
            reconstructed.extend(table_lines[i:])
            break
        
        # Try to collect a complete row
        row_parts = []
        for j in range(columns_per_row):
            if i + j < len(table_lines):
                part = table_lines[i + j].strip()
                if part and not any(word in part.lower() for word in ['subtotal', 'total:', 'tax', 'payment']):
                    row_parts.append(part)
                else:
                    break
        
        # Only create row if we have enough parts and it looks like data
        if len(row_parts) >= 3:
            # Check if it has numbers (indicating it's a data row)
            has_numbers = sum(1 for part in row_parts if re.search(r'\d+', part)) >= 2
            if has_numbers:
                reconstructed.append(' | '.join(row_parts))
                i += len(row_parts)
                continue
        
        # If we couldn't reconstruct, keep as-is
        reconstructed.append(table_lines[i])
        i += 1
    
    return reconstructed