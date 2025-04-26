import pathlib, json, csv, textwrap, pdfplumber, docx, os
from unstructured.partition.auto import partition
import logging
import pandas as pd  # For handling Excel files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_text(path):
    """Extract text from various file formats"""
    try:
        # Handle image files
        if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            logging.info(f"Skipping image file: {path}")
            return "", 0
            
        # Handle Excel files
        if path.suffix.lower() in ['.xlsx', '.xls']:
            logging.info(f"Processing Excel file: {path}")
            try:
                # Read all sheets from the Excel file
                excel_file = pd.ExcelFile(path)
                sheet_names = excel_file.sheet_names
                
                all_text = []
                for sheet in sheet_names:
                    df = pd.read_excel(path, sheet_name=sheet)
                    # Convert each sheet to string and add to all text
                    all_text.append(f"Sheet: {sheet}")
                    all_text.append(df.to_string(index=False))
                
                text = "\n\n".join(all_text)
                # Count sheets as pages - each sheet is considered one page
                return text, len(sheet_names)
            except Exception as e:
                logging.error(f"Error processing Excel file {path}: {str(e)}")
                return "", 0
            
        if path.suffix.lower()==".pdf":
            logging.info(f"Processing PDF: {path}")
            with pdfplumber.open(path) as pdf:
                # Count actual pages from PDF metadata
                page_count = len(pdf.pages)
                return "\n".join(p.extract_text() or "" for p in pdf.pages), page_count
                
        if path.suffix.lower()==".docx":
            logging.info(f"Processing DOCX: {path}")
            try:
                doc = docx.Document(path)
                text = "\n".join(p.text for p in doc.paragraphs)
                
                # More accurate page counting for DOCX:
                # Get document sections to count pages
                sections = len(doc.sections)
                
                # If the document has multiple sections, each section roughly corresponds to a page
                # Otherwise estimate based on paragraphs - roughly 20-25 paragraphs per page
                if sections > 1:
                    page_count = sections
                else:
                    # Count paragraphs with actual content
                    content_paragraphs = sum(1 for p in doc.paragraphs if p.text.strip())
                    page_count = max(1, content_paragraphs // 22)  # ~22 paragraphs per page
                
                return text, page_count
            except Exception as e:
                logging.error(f"Error processing DOCX file {path}: {str(e)}")
                return "", 0
                
        if path.suffix.lower() in [".txt", ".md", ".html"]:
            logging.info(f"Processing text file: {path}")
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                # Estimate page count based on character count for text files
                page_count = max(1, int(len(text) / 3000) + 1)
                return text, page_count
                
        # Fallback to unstructured for other document types
        logging.info(f"Using unstructured fallback for: {path}")
        try:
            elements = partition(str(path))
            if elements and len(elements) > 0:
                text = "\n".join(element.text for element in elements)
                # Estimate page count for other document types
                page_count = max(1, int(len(text) / 3000) + 1)
                return text, page_count
            return "", 0
        except Exception as e:
            logging.error(f"Error processing {path} with unstructured: {str(e)}")
            return "", 0
    except Exception as e:
        logging.error(f"Error processing {path}: {str(e)}")
        return "", 0

rows=[]
for p in pathlib.Path("samples").glob("*"):
    if p.is_file():  # Make sure it's a file
        logging.info(f"Processing: {p}")
        result = extract_text(p)
        
        # Handle the new return format (text, page_count)
        if isinstance(result, tuple) and len(result) == 2:
            txt, page_count = result
        else:
            txt, page_count = "", 0
        
        # Only add files that have content
        if txt:
            rows.append({
                "filename": p.name,
                "bytes": p.stat().st_size,
                "chars": len(txt),
                "tokens": len(txt.split()),
                "pages": page_count,
            })

# Only create a CSV if we have rows
if rows:
    with open("samples/stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "bytes", "chars", "tokens", "pages"])
        # Write header row
        writer.writeheader()
        # Write data rows
        writer.writerows(rows)
    print("Wrote samples/stats.csv with column headers and accurate page counting")
else:
    print("No text was extracted from any files")
