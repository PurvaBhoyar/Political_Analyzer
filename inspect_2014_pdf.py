import pdfplumber

pdf_path = r'c:\Users\DELL\political_analyzer\data\raw\2014 BJP Manifesto Review.pdf'
with pdfplumber.open(pdf_path) as pdf:
    # Check page 5
    page = pdf.pages[5]
    table = page.extract_table()
    if table:
        for row in table[:10]: # Print first 10 rows
            print(row)
    else:
        print("No table found on page 5.")
