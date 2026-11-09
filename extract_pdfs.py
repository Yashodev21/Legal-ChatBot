
import fitz
import os

PDF_FOLDER = "data"
OUTPUT_FOLDER = "extracted_text"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(PDF_FOLDER):

    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, filename)

        document = fitz.open(pdf_path)
        full_text = ""

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            full_text += f"\n\n--- Page {page_number} ---\n\n"
            full_text += text

        output_filename = os.path.splitext(filename)[0] + ".txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(full_text)

        print(f"Extracted: {filename} ({len(document)} pages)")

print("PDF extraction completed!")
