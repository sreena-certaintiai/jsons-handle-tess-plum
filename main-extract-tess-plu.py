import fitz  # PyMuPDF
import os
import shutil
import time
import pandas as pd
import pytesseract
import cv2
import numpy as np
import json
import logging
import pdfplumber
import camelot
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

# Load configuration
with open("config.json", "r") as config_file:
    config = json.load(config_file)

NUM_THREADS = config.get("num_threads", 4)
LOG_LEVEL = config.get("log_level", "INFO").upper()

# Setup logging
LOG_FILE = "pdf_extract.log"
logging.basicConfig(
    handlers=[RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)],
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print(pytesseract.get_tesseract_version())  # Should print the version
# Define folder paths
BASE_DIR = r"storage-tesseract"
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_LBG")
ARCHIVE_FOLDER = os.path.join(BASE_DIR, "archive")
REPORT_FOLDER = os.path.join(BASE_DIR, "report")
IMAGE_FOLDER = os.path.join(BASE_DIR, "image_storage")

# Ensure required folders exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# List to store report data
report_data = []

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF using pdfplumber and Camelot."""
    table_list = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                extracted_table = page.extract_table()
                if extracted_table:
                    table_list.append({"table_id": len(table_list) + 1, "data": extracted_table})

        # If pdfplumber fails, try Camelot
        if not table_list:
            tables = camelot.read_pdf(pdf_path, pages="all")
            for i, table in enumerate(tables):
                table_list.append({"table_id": len(table_list) + 1, "data": table.df.values.tolist()})

        return table_list
    except Exception as e:
        logging.error(f" Error extracting tables: {e}")
        return []

def extract_text_and_images_from_pdf(pdf_path):
    """Extracts text, images, and OCR-based text from a PDF."""
    try:
        doc = fitz.open(pdf_path)
        extracted_text = []
        ocr_extracted_text = []
        image_list = []
        total_images = 0

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                extracted_text.append(text)
            else:
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
                ocr_text = pytesseract.image_to_string(processed_img)

                ocr_extracted_text.append(ocr_text.strip())

            for img_index, img_data in enumerate(page.get_images(full=True)):
                xref = img_data[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_path = os.path.join(IMAGE_FOLDER, f"{os.path.basename(pdf_path).replace('.pdf', '')}_image_{total_images + 1}.png")

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                image_list.append({"image_id": total_images + 1, "image_path": image_path})
                total_images += 1

        return "\n".join(extracted_text), "\n".join(ocr_extracted_text), image_list
    except Exception as e:
        logging.error(f" Error extracting text/images from {pdf_path}: {e}")
        return "", "", []

def process_pdf(filename):
    """Processes a single PDF file and outputs structured JSON."""
    pdf_path = os.path.join(INPUT_FOLDER, filename)
    output_json_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.json")

    file_size_mb = round(os.path.getsize(pdf_path) / (1024 * 1024), 2)

    # Start Time (Before Processing)
    start_datetime = datetime.now()
    start_time_unix = time.time()

    text_output, ocr_output, image_output = extract_text_and_images_from_pdf(pdf_path)
    table_output = extract_tables_from_pdf(pdf_path)

    # Compute Processing Time
    end_time_unix = time.time()
    processing_time = round(end_time_unix - start_time_unix, 2)  # Compute time in seconds

    # Correct End Time Calculation (Fix)
    end_datetime = start_datetime + timedelta(seconds=processing_time)  

    status = "Success" if text_output or ocr_output or table_output or image_output else "Failed"
    failure_reason = "No text, tables, or images found" if status == "Failed" else "N/A"

    # Save as JSON
    json_output = {
        "id": filename,
        "txt_output": text_output,
        "ocr_output": ocr_output,
        "table_output": table_output,
        "image_output": image_output
    }

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_output, json_file, indent=4)

    # Archive processed PDFs
    if status == "Success":
        archive_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_filename = f"{os.path.splitext(filename)[0]}_{archive_timestamp}.pdf"
        archived_path = os.path.join(ARCHIVE_FOLDER, archived_filename)
        shutil.move(pdf_path, archived_path)
        logging.info(f" Moved {filename} -> {archived_filename} in archive")

    # Add to report
    report_data.append([
        filename, file_size_mb, len(text_output.split()), len(ocr_output.split()), len(table_output), len(image_output),
        start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        processing_time, status, failure_reason
    ])

def process_pdfs():
    """Processes all PDFs in parallel using threads and logs total time taken."""
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logging.warning(" No PDF files found in input folder.")
        return

    overall_start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(process_pdf, pdf_files)

    overall_end_time = time.time()
    logging.info(f" Total processing time for all PDFs: {round(overall_end_time - overall_start_time, 2)} seconds")

def generate_report():
    """Generates a CSV report with document processing details."""
    if not report_data:
        logging.warning(" No files were processed, skipping report generation.")
        return

    df = pd.DataFrame(report_data, columns=[
        "Filename", "Size (MB)", "Total Words", "Total OCR Words", "Tables Processed", "Images Extracted",
        "Start Time", "End Time", "Processing Time (Seconds)", "Status", "Failure Reason"
    ])

    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"pdf_extract_{report_timestamp}.csv"
    report_path = os.path.join(REPORT_FOLDER, report_filename)

    df.to_csv(report_path, index=False)
    logging.info(f" Report generated: {report_path}")

if __name__ == "__main__":
    logging.info(" Starting PDF processing...")
    process_pdfs()
    generate_report()
    logging.info(" Processing complete!")
