
# Nourished • GB Health Claims Engine (Streamlit App)

**Authorised for Great Britain only.**  
**Dosage checks disabled by default** (assumes you meet NRV). You can toggle NRV checks on in the sidebar if ever needed.

## Run locally (fastest)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL that Streamlit prints (usually http://localhost:8501).

## How to use
1. In the sidebar, upload the **official GB register .xlsx** (or use the bundled `data/claims_repository.csv`).  
2. (Optional) Upload a synonyms CSV (or use the bundled `data/synonyms.csv`).  
3. Paste/edit your actives in the table (amount/unit optional).  
4. Click **Run Engine** → you'll get **Eligible claims**, **Copy Bank** (with supporting counts), **Theme Counts**.  
5. Click **Download Excel** for a complete workbook output (including the repository and synonyms used).

## Deploy options (quick)
- **Streamlit Community Cloud**: Push these files to a GitHub repo → New app → point to `app.py` → set Python version 3.11+.  
- **Hugging Face Spaces** (Streamlit): New Space → upload files → set SDK to Streamlit → Space runs automatically.

## Notes
- Uses exact GB-approved wording and conditions of use from the register.
- On-hold botanicals are **excluded** by default; you can toggle them in the sidebar.
- The **concise** phrasing in Copy Bank is for internal headline ideation; use the **approved wording** for any external claims.
