import fitz
import os
import glob

pdf_dir = "/vercel/sandbox/uploads"
pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))
pdfs.sort()

for path in pdfs:
    pdf_name = os.path.basename(path)
    doc = fitz.open(path)
    safe_name = pdf_name.replace(".pdf", ".txt")
    # replace problematic chars for filename
    for ch in [" ", "'", "/"]:
        safe_name = safe_name.replace(ch, "_")
    out_path = f"/vercel/sandbox/{safe_name}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"FICHIER: {pdf_name} ({doc.page_count} pages)\n")
        f.write("="*80 + "\n")
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                f.write(f"\n--- PAGE {i+1} ---\n")
                f.write(text)
    doc.close()
    print(f"Extracted: {out_path}")
