"""
Enhanced PDF invoice extractor with text linearization.

This version fixes the major issue with side-by-side layouts by:
1. Using pdfplumber for text extraction
2. Linearizing the text with our custom linearizer 
3. Using proven regex patterns from parser_ocr.py

No more extracting "INVOICE" or dates as company names!
"""

import re
import pdfplumber
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from .linearizer import linearize_text
from ..models import ExtractedInvoice, VendorInfo, CustomerInfo, LineItem

logger = logging.getLogger(__name__)


class InvoiceExtractor:
    """New unified extractor that fixes PDF side-by-side layout issues."""
    
    def __init__(self):
        self.file_path = None
    
    def extract(self, file_path: str) -> ExtractedInvoice:
        """
        Main extraction method with linearization fix.
        
        Process:
        1. Extract text with pdfplumber
        2. Linearize the text (THIS IS THE KEY FIX!)
        3. Use proven regex patterns to extract fields
        """
        self.file_path = file_path
        
        try:
            # Step 1: Extract raw text from PDF
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text or len(text.strip()) < 50:
                logger.warning(f"No sufficient text found in PDF: {file_path}")
                return self._create_empty_invoice()
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            
            # Step 2: LINEARIZE THE TEXT - This fixes everything!
            linearized_text = linearize_text(text)
            logger.info(f"Linearized text: {len(linearized_text)} characters")
            
            # Step 3: Use proven parsing patterns on the linearized text
            return self._parse_with_proven_patterns(linearized_text)
            
        except Exception as e:
            logger.error(f"Error extracting invoice from {file_path}: {str(e)}")
            return self._create_empty_invoice()
    
    def _parse_with_proven_patterns(self, text: str) -> ExtractedInvoice:
        """Parse using the working patterns from parser_ocr.py"""
        
        # Extract all fields using proven patterns
        invoice_number = self._extract_invoice_number(text)
        dates = self._extract_dates(text)
        vendor = self._extract_vendor_info(text)
        customer = self._extract_customer_info(text)
        line_items = self._extract_line_items(text)
        amounts = self._extract_amounts(text)
        currency = self._extract_currency(text)
        
        # Create the invoice object
        invoice = ExtractedInvoice(
            invoice_number=invoice_number,
            invoice_date=dates.get('invoice_date'),
            due_date=dates.get('due_date'),
            vendor=vendor,
            customer=customer,
            line_items=line_items,
            subtotal=amounts.get('subtotal', 0),
            tax_amount=amounts.get('tax', 0),
            total_amount=amounts.get('total', 0),
            currency=currency
        )
        
        # Validate and fix common issues
        return self._validate_and_fix(invoice, text)
    
    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number - enhanced patterns to avoid false matches"""
        patterns = [
            r'Invoice\s*(?:Number|#)\s*:?\s*(INV[-#]?[A-Z0-9\-_]+)',
            r'Invoice\s*#\s*:?\s*([A-Z0-9][A-Z0-9\-_]{2,})',  # At least 3 chars
            r'Invoice\s*No\.?\s*:?\s*([A-Z0-9][A-Z0-9\-_]{2,})',
            r'Bill\s*Number\s*:?\s*([A-Z0-9][A-Z0-9\-_]{2,})',
            r'Document\s*#\s*([A-Z0-9][A-Z0-9\-_]{2,})',
            r'(?:^|\s)INV\s*[-#]\s*([A-Z0-9][A-Z0-9\-_]{2,})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                inv_num = match.group(1).strip()
                # CRITICAL FIX: Skip invalid matches
                if inv_num.upper() not in ['INVOICE', 'ID', '#'] and len(inv_num) >= 3:
                    # Extra validation: shouldn't be a date
                    if not re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', inv_num):
                        return inv_num
        
        return None
    
    def _extract_dates(self, text: str) -> Dict[str, Optional[str]]:
        """Extract invoice and due dates"""
        dates = {'invoice_date': None, 'due_date': None}
        
        # Date patterns
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})',
        ]
        
        # Invoice date
        invoice_patterns = [
            r'Invoice\s*Date\s*:?\s*([^\n]+)',
            r'Date\s*:?\s*([^\n]+)',
            r'Issued\s*:?\s*([^\n]+)',
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_text = match.group(1).strip()
                for date_pattern in date_patterns:
                    date_match = re.search(date_pattern, date_text)
                    if date_match:
                        potential_date = date_match.group(1).strip()
                        if self._validate_date_string(potential_date):
                            dates['invoice_date'] = potential_date
                            break
                if dates['invoice_date']:
                    break
        
        # Due date
        due_patterns = [
            r'Due\s*Date\s*:?\s*([^\n]+)',
            r'Due\s*:?\s*([^\n]+)',
            r'Payment\s*Due\s*:?\s*([^\n]+)',
        ]
        
        for pattern in due_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_text = match.group(1).strip()
                for date_pattern in date_patterns:
                    date_match = re.search(date_pattern, date_text)
                    if date_match:
                        potential_date = date_match.group(1).strip()
                        if self._validate_date_string(potential_date):
                            dates['due_date'] = potential_date
                            break
                if dates['due_date']:
                    break
        
        return dates
    
    def _validate_date_string(self, date_str: str) -> bool:
        """Validate if a date string can be parsed"""
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y',
            '%Y-%m-%d', '%Y/%m/%d',
            '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y'
        ]
        
        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        
        return False
    
    def _extract_vendor_info(self, text: str) -> Optional[VendorInfo]:
        """
        Extract vendor information - CRITICAL FIX for side-by-side issues.
        
        With linearized text, FROM:/TO: sections are now properly separated:
        FROM:
        Company Name
        TO:
        Customer Name
        """
        vendor_info = {}
        
        # Look for vendor indicators - these work much better on linearized text
        vendor_patterns = [
            r'FROM\s*:?\s*\n(.*?)(?=TO\s*:|$)',
            r'Bill\s*From\s*:?\s*\n(.*?)(?=Bill\s*To\s*:|$)',
            r'Vendor\s*:?\s*\n(.*?)(?=Customer\s*:|$)',
            r'Billed\s*By\s*:?\s*\n(.*?)(?=Billed\s*To\s*:|$)',
        ]
        
        for pattern in vendor_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                vendor_text = match.group(1).strip()
                
                # Extract company name (first non-empty line)
                name_lines = [line.strip() for line in vendor_text.split('\n') if line.strip()]
                if name_lines:
                    potential_name = name_lines[0]
                    
                    # CRITICAL: Skip lines that are clearly not company names
                    if self._is_valid_company_name(potential_name):
                        vendor_info['name'] = potential_name
                        
                        # Extract additional info from remaining lines
                        vendor_block = '\n'.join(name_lines[:5])
                        
                        # Extract email
                        email = self._extract_email(vendor_block)
                        if email:
                            vendor_info['email'] = email
                        
                        # Extract phone
                        phone = self._extract_phone(vendor_block)
                        if phone:
                            vendor_info['phone'] = phone
                        
                        # Extract address
                        if len(name_lines) > 1:
                            address_lines = [
                                line for line in name_lines[1:] 
                                if not self._extract_email(line) and not self._extract_phone(line)
                            ]
                            if address_lines:
                                vendor_info['address'] = '\n'.join(address_lines[:3])
                        
                        break
        
        # Fallback: look at the beginning of the document for vendor info
        if not vendor_info.get('name'):
            lines = text.split('\n')
            for i, line in enumerate(lines[:10]):
                line_stripped = line.strip()
                if line_stripped and self._is_valid_company_name(line_stripped):
                    vendor_info['name'] = line_stripped
                    break
        
        if vendor_info.get('name'):
            return VendorInfo(
                name=vendor_info['name'],
                address=vendor_info.get('address'),
                email=vendor_info.get('email'),
                phone=vendor_info.get('phone')
            )
        
        return None
    
    def _is_valid_company_name(self, name: str) -> bool:
        """
        Check if a string is a valid company name.
        
        CRITICAL FUNCTION: This prevents extracting "INVOICE", dates, etc. as company names.
        """
        if not name or len(name) < 2:
            return False
        
        name_upper = name.upper()
        
        # Reject obvious non-company names
        invalid_names = [
            'INVOICE', 'BILL', 'RECEIPT', 'STATEMENT', 'QUOTE',
            'FROM:', 'TO:', 'VENDOR:', 'CLIENT:', 'CUSTOMER:',
            'DESCRIPTION', 'QTY', 'QUANTITY', 'PRICE', 'TOTAL',
            'SUBTOTAL', 'TAX', 'AMOUNT'
        ]
        
        for invalid in invalid_names:
            if invalid in name_upper:
                return False
        
        # Reject dates
        if re.match(r'^\d{1,4}[-/]\d{1,4}[-/]\d{1,4}$', name):
            return False
        
        if re.match(r'^[A-Z]{3,9}\s+\d{1,2},?\s+\d{4}$', name_upper):
            return False
        
        # Reject pure numbers or currency
        if re.match(r'^[\d.,\$€£]+$', name):
            return False
        
        # Must have at least one letter
        if not re.search(r'[A-Za-z]', name):
            return False
        
        # Looks valid
        return True
    
    def _extract_customer_info(self, text: str) -> Optional[CustomerInfo]:
        """Extract customer information from linearized text"""
        customer_info = {}
        
        # Customer patterns - work better on linearized text
        customer_patterns = [
            r'TO\s*:?\s*\n(.*?)(?=\n\s*$|INVOICE|Description)',
            r'Bill\s*To\s*:?\s*\n(.*?)(?=\n\s*$|INVOICE|Description)',
            r'Customer\s*:?\s*\n(.*?)(?=\n\s*$|INVOICE|Description)',
            r'Billed\s*To\s*:?\s*\n(.*?)(?=\n\s*$|INVOICE|Description)',
        ]
        
        for pattern in customer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                customer_text = match.group(1).strip()
                
                # Extract customer name
                name_lines = [line.strip() for line in customer_text.split('\n') if line.strip()]
                if name_lines:
                    potential_name = name_lines[0]
                    
                    if self._is_valid_company_name(potential_name):
                        customer_info['name'] = potential_name
                        
                        # Extract additional info
                        customer_block = '\n'.join(name_lines[:5])
                        
                        email = self._extract_email(customer_block)
                        if email:
                            customer_info['email'] = email
                        
                        phone = self._extract_phone(customer_block)
                        if phone:
                            customer_info['phone'] = phone
                        
                        if len(name_lines) > 1:
                            address_lines = [
                                line for line in name_lines[1:] 
                                if not self._extract_email(line) and not self._extract_phone(line)
                            ]
                            if address_lines:
                                customer_info['address'] = '\n'.join(address_lines[:3])
                        
                        break
        
        if customer_info.get('name'):
            return CustomerInfo(
                name=customer_info['name'],
                address=customer_info.get('address'),
                email=customer_info.get('email'),
                phone=customer_info.get('phone')
            )
        
        return None
    
    def _extract_line_items(self, text: str) -> List[LineItem]:
        """
        Extract line items from linearized text.
        
        The linearizer should have already converted:
        Description
        Qty
        Price  
        Total
        
        Into: Description | Qty | Price | Total
        """
        items = []
        lines = text.split('\n')
        
        # Look for table header first
        table_start = -1
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Check if this line has table headers
            has_desc = 'description' in line_lower
            has_qty = 'qty' in line_lower or 'quantity' in line_lower
            has_price = 'price' in line_lower
            has_total = 'total' in line_lower or 'amount' in line_lower
            
            # If we have at least description and one other column header
            if has_desc and (has_qty or has_price or has_total):
                table_start = i + 1
                break
        
        if table_start < 0:
            logger.warning("No table header found for line items")
            return items
        
        # Process lines after the header
        for i in range(table_start, len(lines)):
            line = lines[i].strip()
            
            if not line:
                continue
            
            # Stop at totals section
            line_lower = line.lower()
            if any(word in line_lower for word in ['subtotal', 'total:', 'tax', 'payment']):
                break
            
            # Try to parse this line as a line item
            item = self._parse_line_item(line)
            if item:
                items.append(item)
        
        return items
    
    def _parse_line_item(self, line: str) -> Optional[LineItem]:
        """
        Parse a single line into a LineItem.
        
        Handles both pipe-separated format (from linearizer) and regular format.
        """
        line = line.strip()
        
        # Strategy 1: Pipe-separated format (from linearizer)
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                try:
                    description = parts[0]
                    qty = self._parse_number(parts[1])
                    price = self._parse_currency(parts[2])
                    total = self._parse_currency(parts[3])
                    
                    # Validate math
                    calculated = qty * price
                    if abs(calculated - total) <= max(total * 0.05, 0.01):
                        return LineItem(
                            description=description,
                            quantity=qty,
                            unit_price=price,
                            total=total
                        )
                except (ValueError, IndexError):
                    pass
        
        # Strategy 2: Regular format with pattern matching
        patterns = [
            # Description Qty Price Total (with optional currency symbols)
            r'^(.+?)\s+(\d+(?:\.\d+)?)\s+\$?([\d,]+(?:\.\d{1,2})?)\s+\$?([\d,]+(?:\.\d{1,2})?)$',
            # Description Qty $Price $Total
            r'^(.+?)\s+(\d+(?:\.\d+)?)\s+\$([\d,]+(?:\.\d{1,2})?)\s+\$([\d,]+(?:\.\d{1,2})?)$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                try:
                    description = match.group(1).strip()
                    qty = float(match.group(2))
                    price = self._parse_currency(match.group(3))
                    total = self._parse_currency(match.group(4))
                    
                    # Validate math
                    calculated = qty * price
                    if abs(calculated - total) <= max(total * 0.05, 0.01):
                        return LineItem(
                            description=description,
                            quantity=qty,
                            unit_price=price,
                            total=total
                        )
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _parse_number(self, text: str) -> float:
        """Parse a number from text"""
        cleaned = re.sub(r'[^\d.]', '', text)
        return float(cleaned) if cleaned else 0.0
    
    def _parse_currency(self, text: str) -> float:
        """Parse currency amount from text"""
        if not text:
            return 0.0
        
        # Remove currency symbols and clean
        cleaned = re.sub(r'[^\d.,]', '', str(text))
        cleaned = cleaned.replace(',', '')
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def _extract_amounts(self, text: str) -> Dict[str, float]:
        """Extract monetary amounts (total, subtotal, tax)"""
        amounts = {}
        
        # Total amount patterns
        total_patterns = [
            r'TOTAL\s*:?\s*\$?([\d,]+\.?\d{0,2})',
            r'Total\s*:?\s*\$?([\d,]+\.?\d{0,2})',
            r'Amount\s*Due\s*:?\s*\$?([\d,]+\.?\d{0,2})',
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amounts['total'] = self._parse_currency(match.group(1))
                break
        
        # Subtotal patterns
        subtotal_patterns = [
            r'Subtotal\s*:?\s*\$?([\d,]+\.?\d{0,2})',
            r'Sub\s*Total\s*:?\s*\$?([\d,]+\.?\d{0,2})',
        ]
        
        for pattern in subtotal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amounts['subtotal'] = self._parse_currency(match.group(1))
                break
        
        # Tax patterns
        tax_patterns = [
            r'Tax\s*:?\s*\$?([\d,]+\.?\d{0,2})',
            r'VAT\s*:?\s*\$?([\d,]+\.?\d{0,2})',
            r'GST\s*:?\s*\$?([\d,]+\.?\d{0,2})',
        ]
        
        for pattern in tax_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amounts['tax'] = self._parse_currency(match.group(1))
                break
        
        return amounts
    
    def _extract_currency(self, text: str) -> str:
        """Extract currency code from text"""
        if 'A$' in text or 'AUD' in text.upper():
            return 'AUD'
        elif 'C$' in text or 'CAD' in text.upper():
            return 'CAD'
        elif '€' in text or 'EUR' in text.upper():
            return 'EUR'
        elif '£' in text or 'GBP' in text.upper():
            return 'GBP'
        elif '$' in text or 'USD' in text.upper():
            return 'USD'
        
        return 'USD'  # Default
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text"""
        phone_patterns = [
            r'\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',
            r'(\d{3})[-.\s](\d{3})[-.\s](\d{4})',
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match and len(match.groups()) == 3:
                return f"({match.group(1)}) {match.group(2)}-{match.group(3)}"
        
        return None
    
    def _validate_and_fix(self, invoice: ExtractedInvoice, text: str) -> ExtractedInvoice:
        """Validate and fix common extraction issues"""
        
        # Fix missing totals if we have line items
        if invoice.line_items and invoice.total_amount == 0:
            calculated_subtotal = sum(item.total for item in invoice.line_items)
            if invoice.subtotal == 0:
                invoice.subtotal = calculated_subtotal
            
            if invoice.tax_amount > 0:
                invoice.total_amount = invoice.subtotal + invoice.tax_amount
            else:
                invoice.total_amount = invoice.subtotal
        
        # Validate vendor name is not invalid
        if invoice.vendor and not self._is_valid_company_name(invoice.vendor.name):
            invoice.vendor.name = "Unknown Vendor"
        
        return invoice
    
    def _create_empty_invoice(self) -> ExtractedInvoice:
        """Create empty invoice with defaults"""
        return ExtractedInvoice(
            vendor=VendorInfo(name="Unknown Vendor"),
            customer=CustomerInfo(name="Unknown Customer"),
            line_items=[],
            total_amount=0
        )