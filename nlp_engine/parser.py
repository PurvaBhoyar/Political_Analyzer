import pdfplumber
import pandas as pd
import os
import re

def extract_review_table(pdf_path):
    """
    Improved PDF Table extractor that handles merged rows and varied columns.
    """
    all_rows = []
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return pd.DataFrame()

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        start_page = min(3, num_pages - 1) if num_pages > 0 else 0
        
        for page in pdf.pages[start_page:]: 
            try:
                table = page.extract_table()
                if not table:
                    continue
                
                for row in table:
                    # Filter out header rows or category rows
                    if row and len(row) >= 3:
                        col0 = str(row[0]) if row[0] else ""
                        promise = str(row[1]).replace('\n', ' ') if row[1] else ""
                        remark = str(row[2]).replace('\n', ' ') if row[2] else ""
                        
                        # If col0 is a number and promise exists, it's a valid data row
                        if (re.search(r'\d+', col0) or len(col0) < 5) and len(promise) > 10:
                            all_rows.append({
                                "promise": promise,
                                "remark": remark
                            })
            except Exception as e:
                print(f"Warning: Failed to extract table from page {page.page_number}: {e}")
                continue
    return pd.DataFrame(all_rows)

def extract_folder_data(folder_path):
    all_data = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found {folder_path}")
        return pd.DataFrame(columns=['promise', 'remark', 'sector'])

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            # Sector name is often after 'Manifesto Document Sheet - '
            sector = file.replace('Manifesto Document Sheet - ', '').replace('.csv', '').strip()
            
            try:
                # Use utf-8 or cp1252 based on actual file encoding
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                # Standardize columns
                df = df.rename(columns={
                    'PROMISE': 'promise',
                    'STATUS AND COMMENTS': 'remark'
                })
                
                if 'promise' in df.columns and 'remark' in df.columns:
                    df['sector'] = sector
                    all_data.append(df[['promise', 'remark', 'sector']])
            except:
                try:
                    df = pd.read_csv(file_path, encoding='cp1252', on_bad_lines='skip')
                    df = df.rename(columns={'PROMISE': 'promise', 'STATUS AND COMMENTS': 'remark'})
                    if 'promise' in df.columns and 'remark' in df.columns:
                        df['sector'] = sector
                        all_data.append(df[['promise', 'remark', 'sector']])
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    
    if not all_data:
        return pd.DataFrame(columns=['promise', 'remark', 'sector'])
        
    final_df = pd.concat(all_data, ignore_index=True)
    # Remove empty rows
    final_df = final_df.dropna(subset=['promise', 'remark'])
    return final_df

def extract_manifesto_promises(pdf_path):
    """
    Extracts promises from a manifesto PDF (non-table format).
    Improved regex to find numbered items or 'We will' sentences.
    """
    all_promises = []
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return pd.DataFrame()

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        start_page = min(5, num_pages - 1) if num_pages > 0 else 0
        
        for page in pdf.pages[start_page:]:
            try:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    current_promise = ""
                    
                    for line in lines:
                        line = line.strip()
                        is_numbered = re.match(r'^\d{1,2}\s+', line)
                        is_we_will = "We will" in line
                        
                        if is_numbered or is_we_will:
                            if current_promise:
                                all_promises.append({"text": current_promise, "page": page.page_number})
                            current_promise = line
                        elif current_promise:
                            current_promise += " " + line
                    
                    if current_promise:
                        all_promises.append({"text": current_promise, "page": page.page_number})
            except Exception as e:
                print(f"Warning: Failed to extract text from page {page.page_number}: {e}")
                continue
                    
    return pd.DataFrame(all_promises)