import pdfplumber
import pandas as pd

def extract_review_table(pdf_path):
    all_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        # Start from Page 4 where the main table begins
        for page in pdf.pages[3:]: 
            table = page.extract_table()
            if table:
                for row in table:
                    # Column 1: Promise text, Column 2: Remarks
                    if row and len(row) >= 3 and row[1] and row[2]:
                        all_rows.append({
                            "promise": row[1].replace('\n', ' '),
                            "remark": row[2].replace('\n', ' ')
                        })
    return pd.DataFrame(all_rows)