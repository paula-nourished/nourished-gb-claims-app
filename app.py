
import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import io
from io import BytesIO
import re

st.set_page_config(page_title="Nourished GB Health Claims Engine", layout="wide")

st.title("Nourished • GB Health Claims Engine")
st.caption("Authorised-for-GB claims only • No dosage checks enforced (assumes NRV is always met)")

with st.sidebar:
    st.header("1) Load GB Register & Synonyms")
    gb_file = st.file_uploader("Upload official GB register (.xlsx)", type=["xlsx"], help="If not provided, the bundled claims_repository.csv will be used")
    syn_file = st.file_uploader("Upload synonyms (.csv)", type=["csv"], help="Optional. Maps name variants to canonical substances.")
    include_on_hold = st.toggle("Include on-hold botanicals", value=False, help="Usually keep this OFF for GB-authorised only")
    enforce_nrv = st.toggle("Enforce ≥15% NRV", value=False, help="OFF by default as requested (no dosage checks)")
    st.markdown("---")
    top_k_themes = st.slider('Themes to feature', 1, 4, 3,
                         help='Controls how many themes appear in the Copy Story (default 3)')
st.markdown('---')
st.header("2) Paste or edit actives")
    st.caption("Amounts/units optional (ignored unless NRV is enforced)")

# Load claims repository
def theme_hint_from_claim(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    theme_map = {
        "immune": "Immunity",
        "fatigue": "Energy",
        "energy": "Energy",
        "cognitive": "Cognitive",
        "psychological": "Cognitive",
        "memory": "Cognitive",
        "hair": "Beauty",
        "skin": "Beauty",
        "nails": "Beauty",
        "bone": "Bone",
        "muscle": "Muscle",
        "vision": "Vision",
        "heart": "Cardio",
        "blood": "Cardio",
        "cholesterol": "Cardio",
        "digest": "Digestive",
        "gut": "Digestive",
    }
    for k, v in theme_map.items():
        if k in t:
            return v
    return ""

def parse_gb_register(uploaded_xlsx) -> pd.DataFrame:
    if uploaded_xlsx is None:
        # load bundled csv
        repo = pd.read_csv("data/claims_repository.csv")
        return repo

    xls = pd.ExcelFile(uploaded_xlsx)
    # Prefer the 'Health_claims' sheet name
    sheet_name = None
    for s in xls.sheet_names:
        if "health" in s.lower():
            sheet_name = s
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
    raw = pd.read_excel(uploaded_xlsx, sheet_name=sheet_name)

    # Map by column index to avoid header text complexity (GB sheet has long headers)
    cols = list(raw.columns)
    # Expected: 0=Claim type, 1=Nutrient/food..., 2=Claim, 3=Conditions..., 4=Health relationship, 5=Scientific Opinion Ref, 6=Regulation, 7=Status in GB, 8=Entry Id
    rename_map = {}
    if len(cols) >= 9:
        rename_map = {
            cols[0]: 'claim_category',
            cols[1]: 'substance_canonical',
            cols[2]: 'claim_text_exact',
            cols[3]: 'conditions_of_use',
            cols[4]: 'health_relationship',
            cols[5]: 'scientific_opinion_reference',
            cols[6]: 'reg_reference',
            cols[7]: 'status_gb',
            cols[8]: 'entry_id',
        }
    raw = raw.rename(columns=rename_map)

    if "status_gb" in raw.columns:
        gb_auth = raw[raw["status_gb"].astype(str).str.strip().str.lower().str.startswith("authorised")].copy()
    else:
        gb_auth = raw.copy()

    for c in ['substance_canonical','claim_text_exact','conditions_of_use','claim_category','reg_reference','health_relationship']:
        if c in gb_auth.columns:
            gb_auth[c] = gb_auth[c].astype(str).str.strip()

    gb_auth["on_hold"] = False

    # Theme hint
    gb_auth["theme_hint"] = gb_auth["claim_text_exact"].apply(theme_hint_from_claim)

    # Claim id
    def make_id(row, i):
        if "entry_id" in row and pd.notna(row["entry_id"]):
            try:
                return f"GB-{int(row['entry_id']):05d}"
            except Exception:
                pass
        base = re.sub(r"[^A-Za-z0-9]+", "-", str(row.get("substance_canonical","")).strip())
        return f"GB-{base.upper()}-{i+1:04d}"

    gb_auth["claim_id"] = [make_id(row, i) for i, row in gb_auth.iterrows()]

    # requires_source_of (informational only; not enforced if enforce_nrv=False)
    def infer_requires_source(text):
        if not isinstance(text, str): return False
        t = text.lower()
        cues = ["source of", "significant amount", "15% nrv", "reference intake", "ri", "rda"]
        return any(c in t for c in cues)
    gb_auth["requires_source_of"] = gb_auth["conditions_of_use"].apply(infer_requires_source)

    # Select standard columns
    cols_out = ["claim_id","substance_canonical","claim_text_exact","conditions_of_use",
                "requires_source_of","on_hold","claim_category","reg_reference","health_relationship","theme_hint"]
    for c in cols_out:
        if c not in gb_auth.columns:
            gb_auth[c] = "" if c not in ["requires_source_of","on_hold"] else False
    repo = gb_auth[cols_out].dropna(subset=["substance_canonical","claim_text_exact"]).reset_index(drop=True)
    return repo

def load_synonyms(uploaded_csv) -> pd.DataFrame:
    if uploaded_csv is None:
        try:
            return pd.read_csv("data/synonyms.csv")
        except Exception:
            return pd.DataFrame(columns=["substance_canonical","synonym"])
    return pd.read_csv(uploaded_csv)

repo = parse_gb_register(gb_file)
syn = load_synonyms(syn_file)

# Filters
if not include_on_hold:
    repo = repo[~repo["on_hold"]]

st.success(f"Loaded {len(repo):,} GB-authorised claims across {repo['substance_canonical'].nunique():,} substances.")

# 3) Actives input table
sample_actives = pd.DataFrame([
    {"ingredient_name":"Beta Glucan","amount_per_day":"","unit":""},
    {"ingredient_name":"Vitamin B1","amount_per_day":"","unit":""},
    {"ingredient_name":"Vitamin C","amount_per_day":"","unit":""},
    {"ingredient_name":"Iron","amount_per_day":"","unit":""},
    {"ingredient_name":"Zinc","amount_per_day":"","unit":""},
    {"ingredient_name":"Vitamin D3","amount_per_day":"","unit":""},
    {"ingredient_name":"Vitamin B6","amount_per_day":"","unit":""},
    {"ingredient_name":"Vitamin B12","amount_per_day":"","unit":""},
], columns=["ingredient_name","amount_per_day","unit"])

actives = st.data_editor(sample_actives, num_rows="dynamic", use_container_width=True)

# Helpers
def canonicalise(name: str, synonyms_df: pd.DataFrame, repo_df: pd.DataFrame) -> str:
    n = (name or "").strip()
    if not n: return n
    low = n.lower()
    if not synonyms_df.empty:
        match = synonyms_df[synonyms_df["synonym"].str.lower() == low]
        if len(match) > 0:
            return match.iloc[0]["substance_canonical"]
    # Exact canonical match
    row = repo_df[repo_df["substance_canonical"].str.lower() == low]
    if len(row) > 0:
        return row.iloc[0]["substance_canonical"]
    # Contains match
    for canon in repo_df["substance_canonical"].unique():
        if canon and canon.lower() in low:
            return canon
    return n

def normalise_claim_core(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.strip()
    low = t.lower()
    if "contributes to" in low:
        part = t[low.index("contributes to") + len("contributes to"):].strip(" .")
        part = re.sub(r"^(the\\s+)?normal\\s+", "", part, flags=re.IGNORECASE)
        return part.strip(" .")
    return t

run = st.button("Run Engine", type="primary")
if run:
    # Build eligible claims (no dosage checks)
    rows = []
    for _, r in actives.iterrows():
        ing = str(r.get("ingredient_name","")).strip()
        if not ing: continue
        canon = canonicalise(ing, syn, repo)
        sub = repo[repo["substance_canonical"].str.lower() == canon.lower()].copy()
        if len(sub) == 0:
            continue
        sub["ingredient_input"] = ing
        sub["amount_input"] = r.get("amount_per_day","")
        sub["unit_input"] = r.get("unit","")
        # If enforce_nrv were ON, we'd compute nrv_pct and set meets_conditions; for now assume True
        sub["nrv_pct"] = np.nan
        sub["meets_conditions"] = True if not enforce_nrv else True  # unchanged per request
        rows.append(sub)

    if rows:
        elig = pd.concat(rows, ignore_index=True)
    else:
        elig = pd.DataFrame(columns=list(repo.columns) + ["ingredient_input","amount_input","unit_input","nrv_pct","meets_conditions"])

    st.subheader("Eligible Claims")
    st.dataframe(elig, use_container_width=True, height=300)

    # Copy Bank
    approved = elig[elig["meets_conditions"]].copy()
    approved["theme"] = approved["claim_text_exact"].apply(theme_hint_from_claim)
    approved["claim_core"] = approved["claim_text_exact"].apply(normalise_claim_core)

    core_counts = approved.groupby("claim_core")["substance_canonical"].nunique().reset_index(name="supporting_ingredient_count")
    supporters = approved.groupby("claim_core")["substance_canonical"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index(name="supported_by")
    exemplar = approved.groupby("claim_core")["claim_text_exact"].first().reset_index(name="approved_wording_example")
    theme_per_core = (approved.groupby(["claim_core","theme"]).size()
                      .reset_index(name="n").sort_values(["claim_core","n"], ascending=[True, False])
                      .drop_duplicates("claim_core")[["claim_core","theme"]])
    copy_bank = exemplar.merge(core_counts, on="claim_core").merge(supporters, on="claim_core").merge(theme_per_core, on="claim_core")
    copy_bank["concise_variation_(check_compliance)"] = copy_bank["claim_core"].apply(lambda x: re.sub(r"\\bfunction of (the )?\\b","", x or "", flags=re.IGNORECASE).strip().capitalize())

    st.subheader("Copy Bank (grouped cores + counts)")
    st.dataframe(copy_bank[["theme","claim_core","approved_wording_example","supported_by","supporting_ingredient_count","concise_variation_(check_compliance)"]],
                 use_container_width=True, height=300)

    theme_counts = copy_bank.groupby("theme")["supporting_ingredient_count"].sum().reset_index(name="total_supporting_ingredients").sort_values("total_supporting_ingredients", ascending=False)
    st.subheader("Theme Counts")
    st.dataframe(theme_counts, use_container_width=True, height=240)

    # ---------- Copy Story (headline + merged-claims sentences) ----------
def human_join(items):
    items = [x for x in items if str(x).strip()]
    if not items: return ""
    if len(items) == 1: return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]

def phrase_for_theme(theme: str) -> str:
    m = {
        "Immunity": "immunity",
        "Energy": "energy",
        "Cognitive": "cognitive function",
        "Beauty": "skin, hair and nails",
        "Bone": "bone health",
        "Muscle": "muscle function",
        "Vision": "vision",
        "Cardio": "heart and blood health",
        "Digestive": "digestive health",
    }
    return m.get(theme, (theme or "").lower())

def after_contributes_to(approved_text: str) -> str:
    if not isinstance(approved_text, str): return ""
    low = approved_text.lower()
    key = "contributes to"
    i = low.find(key)
    return approved_text.strip() if i == -1 else approved_text[i+len(key):].strip()

# Top-K themes for the story
top_themes = theme_counts.sort_values("total_supporting_ingredients", ascending=False)["theme"].tolist()[:top_k_themes]

# Headline
headline = f"Precision nutrition formulated to support {human_join([phrase_for_theme(t) for t in top_themes])}." if top_themes else ""

# Build merged sentences for each chosen theme
merged_sentences = []
for t in top_themes:
    cb_t = copy_bank[copy_bank["theme"] == t].copy()
    if cb_t.empty: 
        continue
    cb_t = cb_t.sort_values("supporting_ingredient_count", ascending=False)
    row = cb_t.iloc[0]
    core = row["claim_core"]
    approved_text = row["approved_wording_example"]
    subs = (approved[approved["claim_text_exact"] == approved_text]["substance_canonical"]
            .dropna().drop_duplicates().tolist())
    if not subs:
        subs = (approved[approved["claim_core"] == core]["substance_canonical"]
                .dropna().drop_duplicates().tolist())
    subs_sorted = sorted(subs, key=lambda x: x.lower())
    subject = human_join(subs_sorted)
    verb = "contributes" if len(subs_sorted) == 1 else "contribute"
    tail = after_contributes_to(approved_text)
    sentence = f"{subject} {verb} to {tail}"
    if not sentence.endswith(('.', '!', '?')):
        sentence += "."
    merged_sentences.append({"theme": t, "sentence": sentence, "supporting_ingredient_count": len(subs_sorted)})

st.subheader("Copy Story")
if headline:
    st.markdown(f"### {headline}")

# Simple JS copy button
def copy_button(label: str, text_to_copy: str, key: str):
    if not isinstance(text_to_copy, str):
        text_to_copy = ""
    esc = (text_to_copy.replace("\\", "\\\\")
                      .replace("'", "\\'")
                      .replace("\n", "\\n"))
    html(f"""
        <button onclick="navigator.clipboard.writeText('{esc}')"
                style="margin:2px 0;padding:6px 10px;border-radius:6px;border:1px solid #ddd;background:#fafafa;cursor:pointer;">
            {label}
        </button>
    """, height=40, key=key)

if headline:
    copy_button('Copy headline', headline, 'copy_headline')

if merged_sentences:
    st.write("**Merged claims (auto-built from GB-authorised wording):**")
    for idx, s in enumerate(merged_sentences):
        st.markdown(f"- {s['sentence']}")
        copy_button('Copy line', s['sentence'], f'copy_line_{idx}')

# Combined paragraph variant
copy_paragraph = (headline + " " if headline else "") + " ".join([s["sentence"] for s in merged_sentences])
if copy_paragraph:
    copy_button('Copy full paragraph', copy_paragraph, 'copy_full_para')

# Prepare Copy_Story sheet for export
copy_story_df = pd.DataFrame(
    [{"headline": headline}] +
    [{"theme": s["theme"], "sentence": s["sentence"], "supporting_ingredient_count": s["supporting_ingredient_count"]}
     for s in merged_sentences]
)

    # Excel export
    def to_excel_bytes(elig_df, cb_df, tc_df, repo_df, syn_df, copy_story_df, copy_story_variants_df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            elig_df.to_excel(writer, sheet_name="eligible_claims", index=False)
            cb_df.to_excel(writer, sheet_name="Copy_Bank", index=False)
            tc_df.to_excel(writer, sheet_name="Theme_Counts", index=False)
            copy_story_df.to_excel(writer, sheet_name="Copy_Story", index=False)
            copy_story_variants_df.to_excel(writer, sheet_name="Copy_Story_Variants", index=False)
            repo_df.to_excel(writer, sheet_name="claims_repository", index=False)
            syn_df.to_excel(writer, sheet_name="synonyms", index=False)
        output.seek(0)
        return output.getvalue()

    copy_story_variants_df = pd.DataFrame({
    'variant': ['headline','paragraph'],
    'text': [headline, copy_paragraph]
})
xbytes = to_excel_bytes(elig, copy_bank, theme_counts, repo, syn, copy_story_df, copy_story_variants_df)
    st.download_button("Download Excel", data=xbytes, file_name="claims_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Tip: add more synonyms so ingredient names always map to the right canonical substance (e.g., 'Vit C' → 'Vitamin C').")
