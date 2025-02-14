import os
import subprocess
import sys
import time
from pathlib import Path
import base64
import io
from tqdm import tqdm

# Auto-install required packages
def install_requirements():
    packages = [
        'PyMuPDF',
        'google-generativeai',
        'pillow',
        'tqdm'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("Installing required packages...")
install_requirements()

import fitz
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# API Configuration
GOOGLE_API_KEY = "AIXXXXXXXXXXXXXXXXXXXXXXXX"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Configure the model
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif'}

def ocr_with_retry(image_data, retries=3):
    """Perform OCR with retry mechanism using Gemini Vision API."""
    for attempt in range(retries):
        try:
            # Create the prompt with the image
            response = model.generate_content([
                {
                    "mime_type": "image/png",
                    "data": image_data
                },
                "Please extract and organize the content in a clean, readable format with proper line break and organization:\n\n"
                "1. Keep essential information:\n"
                "   - Title and Subtitle if any\n"
                "   - Main content (like poetry or text)\n"
                "   - Page Number and Book Name in Top Right if any\n"
                "   - Author information if any\n"
                "   - Source attributions if any\n"
                "   - Section titles or headers if any\n\n"
                "   - Table-like structures if any\n\n"
                "2. Remove common distracting elements:\n"
                "   - Annotation marks\n"
                "   - Commentary\n"
                "   - Reference numbers\n"
                "   - Editorial notes\n\n"
                "Start your response with ''' on a new line, then provide the organized content, then end with ''' on a new line."
            ])
            
            if response and hasattr(response, 'text'):
                return response.text
            else:
                raise Exception(f"Invalid response from Gemini API")
            
        except Exception as e:
            if attempt < retries - 1:
                print(f"\nAttempt {attempt + 1} failed. Retrying in 5 seconds... Error: {str(e)}")
                time.sleep(5)
            else:
                print(f"\nFinal attempt failed: {str(e)}")
                return None

def process_image(image_path, output_dir):
    """Process a single image file with OCR."""
    try:
        # Open and convert image
        img = Image.open(image_path)
        img = img.convert('RGB')
        
        # Handle image dimensions
        width, height = img.size
        if width < 768 or height < 768:
            # Scale up small images to Gemini's recommended minimum
            scale_factor = max(768/width, 768/height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        elif width > 3072 or height > 3072:
            # Scale down large images to Gemini's recommended maximum
            scale_factor = min(3072/width, 3072/height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save temporary optimized image
        temp_image = output_dir / f"temp_{Path(image_path).name}"
        img.save(temp_image, "PNG", optimize=True)
        
        # Convert to base64
        with open(temp_image, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Call OCR API
        ocr_text = ocr_with_retry(image_data)
        
        # Clean up
        temp_image.unlink(missing_ok=True)
        
        return ocr_text if ocr_text else None
        
    except Exception as e:
        print(f"\nError processing image {image_path}: {str(e)}")
        return None

def process_pdf(pdf_path):
    """Process a single PDF file with OCR."""
    pdf_name = Path(pdf_path).stem
    output_dir = Path(pdf_name)
    output_dir.mkdir(exist_ok=True)
    
    # Skip if already processed
    if Path(f"{pdf_name}_complete.md").exists():
        print(f"Skipping {pdf_path} - already processed")
        return
    
    print(f"\nProcessing {pdf_path}...")
    
    try:
        # Open PDF and get page count
        pdf_doc = fitz.open(pdf_path)
        total_pages = pdf_doc.page_count
        
        # Create progress bar for pages
        pbar = tqdm(total=total_pages, desc="Pages processed", unit="page")
        
        processed_pages = []
        for page_num in range(total_pages):
            page_md = output_dir / f"{pdf_name}_page_{page_num + 1}.md"
            
            if page_md.exists():
                print(f"\nSkipping page {page_num + 1} - already processed")
                processed_pages.append(page_md.read_text(encoding="utf-8"))
                pbar.update(1)
                continue
            
            try:
                # Convert PDF page to image
                page = pdf_doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                img = img.convert('RGB')
                
                # Save temporary page image
                temp_image_path = output_dir / f"temp_page_{page_num + 1}.png"
                
                # Handle image dimensions
                width, height = img.size
                if width < 768 or height < 768:
                    scale_factor = max(768/width, 768/height)
                    new_size = (int(width * scale_factor), int(height * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                elif width > 3072 or height > 3072:
                    scale_factor = min(3072/width, 3072/height)
                    new_size = (int(width * scale_factor), int(height * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save the temporary image
                img.save(temp_image_path, "PNG", optimize=True)
                
                # Convert to base64
                with open(temp_image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Call OCR API
                ocr_text = ocr_with_retry(image_data)
                
                if ocr_text:
                    # Ensure we're writing a string
                    if isinstance(ocr_text, (list, dict)):
                        ocr_text = str(ocr_text)
                    
                    with open(page_md, "w", encoding="utf-8") as f:
                        f.write(ocr_text)
                    processed_pages.append(ocr_text)
                
                # Clean up temporary image
                temp_image_path.unlink(missing_ok=True)
                
                # Update progress
                pbar.update(1)
                
                # Add delay between pages to respect rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"\nError processing page {page_num + 1}: {str(e)}")
                pbar.update(1)
        
        pbar.close()
        
        # Combine successful pages
        if processed_pages:
            print("\nCreating combined markdown file...")
            with open(f"{pdf_name}_complete.md", "w", encoding="utf-8") as f:
                f.write("\n\n---\n\n".join(processed_pages))
            print(f"Created: {pdf_name}_complete.md")
        
    except Exception as e:
        print(f"\nError processing PDF: {str(e)}")
    finally:
        if 'pdf_doc' in locals():
            pdf_doc.close()

def process_single_image_file(image_path):
    """Process a single image file."""
    image_name = Path(image_path).stem
    output_dir = Path("image_ocr")
    output_dir.mkdir(exist_ok=True)
    
    # Skip if already processed
    output_file = output_dir / f"{image_name}.md"
    if output_file.exists():
        print(f"Skipping {image_path} - already processed")
        return
    
    print(f"\nProcessing image {image_path}...")
    
    ocr_text = process_image(image_path, output_dir)
    if ocr_text:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(ocr_text)
        print(f"Created: {output_file}")

def main():
    """Process all PDFs and supported image files in current directory."""
    # Get all PDF and image files
    pdf_files = list(Path('.').glob('*.pdf'))
    image_files = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        image_files.extend(Path('.').glob(f'*{ext}'))
    
    if not pdf_files and not image_files:
        print("No PDF or supported image files found in current directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files and {len(image_files)} image files")
    
    # Process PDFs
    if pdf_files:
        print("\nProcessing PDF files:")
        with tqdm(pdf_files, desc="PDF files processed", unit="file") as pbar:
            for pdf_file in pbar:
                process_pdf(pdf_file)
                pbar.set_description(f"Processing {pdf_file.name}")
                time.sleep(2)  # Add delay between files
    
    # Process Images
    if image_files:
        print("\nProcessing image files:")
        with tqdm(image_files, desc="Image files processed", unit="file") as pbar:
            for image_file in pbar:
                process_single_image_file(image_file)
                pbar.set_description(f"Processing {image_file.name}")
                time.sleep(2)  # Add delay between files

if __name__ == "__main__":
    main()