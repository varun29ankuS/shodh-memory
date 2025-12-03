# JPEG to PDF Converter
## Quick Instructions

---

## Method 1: Easy Way (Double-Click)

1. **Place your 4 JPEG files** in the `docs/` folder
   - Any JPEG files (.jpg, .jpeg, .JPG, .JPEG)

2. **Double-click** `convert_to_pdf.bat`
   - It will automatically:
     - Install Pillow (if needed)
     - Find all JPEGs
     - Convert to `submission_document.pdf`
     - Open the PDF

3. **Done!** Your PDF is ready at:
   ```
   docs/submission_document.pdf
   ```

---

## Method 2: Command Line (Custom Names)

### Install Pillow first (one-time):
```bash
pip install Pillow
```

### Convert specific 4 images:
```bash
cd docs/
python convert_images_to_pdf.py image1.jpg image2.jpg image3.jpg image4.jpg -o my_document.pdf
```

### Convert all JPEGs in folder:
```bash
python convert_images_to_pdf.py *.jpg -o output.pdf
```

### Options:
```bash
# Keep original order (don't sort alphabetically)
python convert_images_to_pdf.py img1.jpg img2.jpg img3.jpg img4.jpg -o doc.pdf --no-sort

# Custom output name
python convert_images_to_pdf.py *.jpg -o drone_challenge_submission.pdf
```

---

## Method 3: Online (No Installation)

If Python isn't working, use these free online tools:

1. **iLovePDF**: https://www.ilovepdf.com/jpg_to_pdf
   - Drag & drop 4 JPEGs
   - Click "Convert to PDF"
   - Download

2. **PDF24**: https://tools.pdf24.org/en/images-to-pdf
   - Upload images
   - Arrange order
   - Create PDF

3. **Smallpdf**: https://smallpdf.com/jpg-to-pdf
   - Upload JPEGs
   - Combine into one PDF

---

## Troubleshooting

### "Python not found"
**Solution**: Install Python from https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### "No module named PIL"
**Solution**: Install Pillow
```bash
pip install Pillow
```

### "No JPEG files found"
**Solution**: Make sure your JPEG files are in the same folder as the scripts

### Images in wrong order
**Solution**: Rename images to sort properly:
```
01_diagram.jpg
02_screenshot.jpg
03_benchmark.jpg
04_architecture.jpg
```

Or use `--no-sort` and specify order manually:
```bash
python convert_images_to_pdf.py 01.jpg 02.jpg 03.jpg 04.jpg -o output.pdf --no-sort
```

---

## For Drone Challenge Submission

Suggested naming for your 4 images:
1. `01_system_architecture.jpg` - System block diagram
2. `02_api_demo.jpg` - API request/response
3. `03_benchmark_results.jpg` - 100% accuracy proof
4. `04_graph_visualization.jpg` - Knowledge graph

Then run:
```bash
python convert_images_to_pdf.py 01_*.jpg 02_*.jpg 03_*.jpg 04_*.jpg -o drone_challenge_diagrams.pdf --no-sort
```

---

## What the Script Does

1. ✅ Validates all image files exist
2. ✅ Converts images to RGB (required for PDF)
3. ✅ **Adds filename as caption under each image** (NEW!)
4. ✅ Maintains image quality (95% quality)
5. ✅ Creates multi-page PDF (one image per page)
6. ✅ Shows progress and file size
7. ✅ Supports JPEG and PNG formats

## Caption Feature

**By default, the filename will appear centered below each image in the PDF!**

For example:
- Image file: `01_system_architecture.jpg`
- PDF will show: Image with "01_system_architecture.jpg" centered below it

### Turn off captions (if you don't want them):
```bash
python convert_images_to_pdf.py *.jpg -o output.pdf --no-captions
```

---

## Example Output

```
📄 Converting 4 images to PDF...
   Adding filenames as captions...
  ✓ Loaded: 01_system_architecture.jpg (1920x1080)
  ✓ Loaded: 02_api_demo.jpg (1600x900)
  ✓ Loaded: 03_benchmark_results.jpg (1920x1080)
  ✓ Loaded: 04_graph_visualization.jpg (1920x1080)

✅ PDF created successfully!
   Output: drone_challenge_diagrams.pdf
   Pages: 4
   Size: 2.34 MB
```

**Each page in the PDF will look like:**
```
┌─────────────────────────────┐
│                             │
│     [Your Image Here]       │
│                             │
│                             │
├─────────────────────────────┤
│  01_system_architecture.jpg │  ← Caption added automatically!
└─────────────────────────────┘
```

---

## Files Created

- `convert_images_to_pdf.py` - Python script (cross-platform)
- `convert_to_pdf.bat` - Windows batch file (double-click)
- `JPEG_TO_PDF_INSTRUCTIONS.md` - This file

---

**Need help?** Just double-click `convert_to_pdf.bat` - it handles everything automatically!
