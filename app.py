import streamlit as st
import altair as alt
import pandas as pd
import py3Dmol
import json
import tempfile
import os
from fpdf import FPDF
from streamlit_molstar import st_molstar
from src.logic_engine import *
from src.agent import load_overview_text, run_groq_agent, DEFAULT_MODEL

# NOTE: Storing API keys in code is insecure. Prefer env vars or Streamlit secrets.
GROQ_API_KEY = "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"
try:
    import vl_convert as vl
except Exception:
    vl = None

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

def parse_secondary_structure(pdb_text: str):
    ss_map = {}
    for line in pdb_text.splitlines():
        if line.startswith("HELIX"):
            try:
                chain = line[19].strip()
                start = int(line[21:25].strip())
                end = int(line[33:37].strip())
                if chain in ("", "A"):
                    for r in range(start, end + 1):
                        ss_map[r] = "H"
            except Exception:
                continue
        elif line.startswith("SHEET"):
            try:
                chain = line[21].strip()
                start = int(line[22:26].strip())
                end = int(line[33:37].strip())
                if chain in ("", "A"):
                    for r in range(start, end + 1):
                        if r not in ss_map:
                            ss_map[r] = "S"
            except Exception:
                continue
    return ss_map

def color_by_plddt(plddt):
    if plddt is None:
        return "#ffffff"
    if plddt >= 90:
        return "#1f77b4"
    if plddt >= 70:
        return "#7fc7ff"
    if plddt >= 50:
        return "#ffbf00"
    return "#d62728"

def color_by_secondary(ss_type):
    if ss_type == "H":
        return "#ff7f0e"
    if ss_type == "S":
        return "#2ca02c"
    return "#E0E0E0"

def color_by_hydro(aa1):
    aa3 = AA1_TO_AA3.get(aa1)
    val = HYDRO_MAP.get(aa3, 0.0) if aa3 else 0.0
    return "#ff0000" if val > 0 else "#0000ff"

def assign_plddt_band(plddt: float) -> str:
    if plddt >= 90:
        return "Very high (pLDDT >= 90)"
    if plddt >= 70:
        return "High (70 <= pLDDT < 90)"
    if plddt >= 50:
        return "Low (50 <= pLDDT < 70)"
    return "Very low (pLDDT < 50)"

def build_report(uid: str, protein_df: pd.DataFrame, hotspots: pd.DataFrame) -> dict:
    gii_score = float(protein_df["am_pathogenicity"].mean()) if not protein_df.empty else 0.0
    num_clusters = (
        int(hotspots[hotspots["cluster_id"] >= 0]["cluster_id"].nunique())
        if not hotspots.empty
        else 0
    )
    if gii_score >= 0.75:
        status = "CRITICAL"
    elif gii_score >= 0.55:
        status = "WARNING"
    else:
        status = "STABLE"
    return {
        "uniprot_id": uid,
        "instability_index": gii_score,
        "anomaly_clusters": num_clusters,
        "status": status,
    }

def altair_chart_to_png(chart: alt.Chart) -> bytes:
    if vl is None:
        return None
    spec = chart.to_dict()
    return vl.vegalite_to_png(spec)

def report_to_pdf(report: dict, plot_pngs=None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Rare-Disease Structural Anomaly Report", ln=True)
    pdf.ln(4)
    for key, value in report.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)

    if plot_pngs:
        pdf.ln(6)
        for title, png_bytes in plot_pngs:
            if not png_bytes:
                continue
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8, title, ln=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(png_bytes)
                tmp_path = tmp.name
            try:
                pdf.image(tmp_path, w=190)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    pdf_bytes = pdf.output(dest="S")
    return bytes(pdf_bytes)

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Anomaly Detector PRO", layout="wide", page_icon="🧬")

# --- 2. STYLING ---
st.markdown("""
<style>
    .seq-container {
        overflow-x: auto;
        white-space: nowrap;
        background: #f7f7f9;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e4e6eb;
        font-family: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
        color: #111827;
        margin-bottom: 20px;
    }
    .res-item {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 18px;
        height: 28px;
        margin: 1px;
        border-radius: 4px;
        font-size: 12px;
        line-height: 1;
        cursor: default;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .res-index { font-size: 8px; color: #6b7280; line-height: 1; }
    .res-item:hover { outline: 1px solid #111827; }
</style>
""", unsafe_allow_html=True)

# --- 3. BI-DIRECTIONAL SYNC LOGIC ---
if 'd_to_g' not in st.session_state:
    st.session_state.d_to_g, st.session_state.g_to_d = load_rare_disease_catalog("data/references/en_product6.xml")

def sync_gene(): st.session_state.gene_input = st.session_state.d_to_g.get(st.session_state.disease_select, "SOD1")
def sync_disease(): 
    gene = st.session_state.gene_input.upper().strip()
    st.session_state.disease_select = st.session_state.g_to_d.get(gene, list(st.session_state.d_to_g.keys())[0])

# --- 4. SIDEBAR GATEWAY ---
st.sidebar.header("🔍 Search Gateway")
st.sidebar.selectbox("Select Rare Disease", options=list(st.session_state.d_to_g.keys()), key="disease_select", on_change=sync_gene)
gene_symbol = st.sidebar.text_input("Gene Symbol", key="gene_input", on_change=sync_disease, value="SOD1").upper().strip()

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Structural Layers")
show_hydro = st.sidebar.toggle("Hydrophobicity Map")
show_sec = st.sidebar.toggle("Secondary Structure")
show_conf = st.sidebar.toggle("Model Confidence (pLDDT)")
show_hotspots = st.sidebar.toggle("AI Hotspot Overlay", value=True)

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
run_btn = st.sidebar.button("Run Global Analysis", type="primary", use_container_width=True)
if run_btn:
    st.session_state.run_analysis = True

# --- 5. MAIN UI ---
st.title("Rare-Disease Structural Anomaly Detector")
st.caption(f"Target: {gene_symbol} | {st.session_state.disease_select}")

overview_text = load_overview_text("data/knowledge/overview.txt")

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

if st.session_state.run_analysis:
    with st.spinner("Analyzing..."):
        uid = resolve_uniprot_id(gene_symbol)
        if not uid: st.error("ID resolution failed.")
        else:
            am_file = "data/raw/AlphaMissense_hg38.tsv.gz"
            protein_df = get_protein_anomalies_cached(uid, am_file)
            local_pdb = download_pdb_locally(get_af_structure_url(uid), uid)
            
            if not protein_df.empty and local_pdb:
                # 1. HOTSPOTS + PDB TEXT
                high_risk = protein_df[protein_df['am_pathogenicity'] > 0.6]
                hotspots = detect_structural_hotspots(local_pdb, high_risk)
                with open(local_pdb, 'r') as f: pdb_text = f.read()

                # 2. SEQUENCE RIBBON
                plddt_df = extract_plddt_from_pdb(local_pdb)
                plddt_map = dict(zip(plddt_df['residue_num'], plddt_df['plddt'])) if not plddt_df.empty else {}
                seq = get_sequence_from_pdb(local_pdb)
                st.subheader(f"Amino Acid Sequence (Length: {len(seq)})")
                seq_html = '<div class="seq-container">'
                ss_map = parse_secondary_structure(pdb_text)
                hotspot_set = set(hotspots[hotspots['cluster_id'] >= 0]['residue_num'].tolist()) if show_hotspots and not hotspots.empty else set()
                if show_conf:
                    seq_mode = "conf"
                elif show_sec:
                    seq_mode = "sec"
                elif show_hydro:
                    seq_mode = "hydro"
                else:
                    seq_mode = "none"
                for i, aa in enumerate(seq):
                    res_num = i + 1
                    if res_num in hotspot_set:
                        bg = "#ff2d55"
                    elif seq_mode == "conf":
                        bg = color_by_plddt(plddt_map.get(res_num, None))
                    elif seq_mode == "sec":
                        bg = color_by_secondary(ss_map.get(res_num))
                    elif seq_mode == "hydro":
                        bg = color_by_hydro(aa)
                    else:
                        bg = "#ffffff"
                    seq_html += f'<div class="res-item" style="background:{bg};"><span class="res-index">{res_num}</span>{aa}</div>'
                seq_html += '</div>'
                st.markdown(seq_html, unsafe_allow_html=True)

                # 3. 3D SIDE-BY-SIDE
                
                v_col1, v_col2 = st.columns(2)

                with v_col1:
                    st.caption("Protein Structure")
                    st_molstar(local_pdb, key=f"mol_{uid}", height=500)

                with v_col2:
                    if show_conf:
                        right_title = "Model Confidence (pLDDT)"
                    elif show_sec:
                        right_title = "Secondary Structure"
                    elif show_hydro:
                        right_title = "Hydrophobicity Map"
                    else:
                        right_title = "Anomaly Overlay"
                    st.caption(right_title)
                    view2 = py3Dmol.view(width=500, height=500); view2.addModel(pdb_text, "pdb")
                    if show_conf and not plddt_df.empty:
                        bands = [
                            (90, 101, "#1f77b4"),
                            (70, 90, "#7fc7ff"),
                            (50, 70, "#ffbf00"),
                            (0, 50, "#d62728"),
                        ]
                        for low, high, color in bands:
                            res = plddt_df[(plddt_df['plddt'] >= low) & (plddt_df['plddt'] < high)]['residue_num'].tolist()
                            view2.setStyle({"resi": res}, {"cartoon": {"color": color}})
                    elif show_hydro:
                        for res_name, val in HYDRO_MAP.items():
                            view2.setStyle({"resn": res_name}, {"cartoon": {"color": "#ff0000" if val > 0 else "#0000ff"}})
                    elif show_sec:
                        view2.setStyle({"ss": "h"}, {"cartoon": {"color": "#ff7f0e"}})
                        view2.setStyle({"ss": "s"}, {"cartoon": {"color": "#2ca02c"}})
                    else: view2.setStyle({}, {"cartoon": {"color": "#E0E0E0"}})
                    
                    if show_hotspots and not hotspots.empty:
                        res_list = hotspots[hotspots['cluster_id'] >= 0]['residue_num'].tolist()
                        view2.addStyle({"resi": res_list}, {"stick": {"color": "#ff2d55", "radius": 0.8}})
                        view2.addSurface(py3Dmol.VDW, {"opacity": 0.5, "color": "#ff2d55"}, {"resi": res_list})
                    view2.zoomTo(); st.components.v1.html(view2._make_html(), height=500)


                # 3. ANALYTICS & LEDGER
                st.markdown("---")
                st.subheader("📋 Predicted Genomic Anomalies")
                st.dataframe(protein_df[['variant', 'residue_num', 'am_pathogenicity', 'am_class']].sort_values(by='am_pathogenicity', ascending=False), use_container_width=True, hide_index=True)

                chart_data = (
                    protein_df.groupby('residue_num')['am_pathogenicity']
                    .mean()
                    .reset_index()
                )
                line_chart = (
                    alt.Chart(chart_data)
                    .mark_area(
                        line={"color": "darkred"},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="white", offset=0),
                                alt.GradientStop(color="red", offset=1),
                            ],
                            x1=1,
                            x2=1,
                            y1=1,
                            y2=0,
                        ),
                    )
                    .encode(
                        x=alt.X("residue_num:Q", title="Residue Position"),
                        y=alt.Y("am_pathogenicity:Q", title="Anomaly Score"),
                        tooltip=["residue_num", "am_pathogenicity"],
                    )
                    .interactive()
                )

                conf_chart = None
                plddt_plot_df = None
                if not plddt_df.empty:
                    plddt_plot_df = plddt_df.copy()
                    plddt_plot_df["band"] = plddt_plot_df["plddt"].apply(assign_plddt_band)
                    conf_chart = (
                        alt.Chart(plddt_plot_df)
                        .mark_line()
                        .encode(
                            x=alt.X("residue_num:Q", title="Residue Position"),
                            y=alt.Y(
                                "plddt:Q",
                                title="pLDDT (0-100)",
                                scale=alt.Scale(domain=[0, 100]),
                            ),
                            color=alt.Color(
                                "band:N",
                                scale=alt.Scale(
                                    domain=[
                                        "Very high (pLDDT >= 90)",
                                        "High (70 <= pLDDT < 90)",
                                        "Low (50 <= pLDDT < 70)",
                                        "Very low (pLDDT < 50)",
                                    ],
                                    range=["#1f77b4", "#2ca02c", "#ffbf00", "#d62728"],
                                ),
                                legend=alt.Legend(title="Confidence Band"),
                            ),
                            tooltip=["residue_num", "plddt", "band"],
                        )
                        .interactive()
                    )

                # 4. REPORT
                report = build_report(uid, protein_df, hotspots)
                r1, r2, r3 = st.columns(3)
                r1.metric("Clinical Status", report["status"])
                r2.metric("Instability Index", f"{report['instability_index']:.3f}")
                r3.metric("3D Hotspots", report["anomaly_clusters"])

                # Build context for agent
                top_anomalies = (
                    protein_df.sort_values(by="am_pathogenicity", ascending=False)
                    .head(5)[["variant", "residue_num", "am_pathogenicity"]]
                    .to_dict(orient="records")
                )
                hotspot_residues = (
                    hotspots[hotspots["cluster_id"] >= 0]["residue_num"].unique().tolist()
                    if not hotspots.empty
                    else []
                )
                st.session_state.agent_context = (
                    f"Gene: {gene_symbol}\n"
                    f"Disease: {st.session_state.disease_select}\n"
                    f"UniProt ID: {uid}\n"
                    f"Instability Index (mean pathogenicity): {report['instability_index']:.3f}\n"
                    f"Clinical Status: {report['status']}\n"
                    f"Hotspot clusters: {report['anomaly_clusters']}\n"
                    f"Top anomalies: {top_anomalies}\n"
                    f"Hotspot residues (sample): {hotspot_residues[:20]}\n"
                )

                st.subheader("Export Report")
                plot_pngs = []
                try:
                    plot_pngs.append(("Instability Plot (2D)", altair_chart_to_png(line_chart)))
                except Exception:
                    plot_pngs.append(("Instability Plot (2D)", None))
                if conf_chart is not None:
                    try:
                        plot_pngs.append(("Model Confidence (pLDDT)", altair_chart_to_png(conf_chart)))
                    except Exception:
                        plot_pngs.append(("Model Confidence (pLDDT)", None))
                st.download_button(
                    "Download JSON",
                    data=json.dumps(report, indent=2),
                    file_name=f"{uid}_anomaly_report.json",
                    mime="application/json",
                )
                st.download_button(
                    "Download PDF",
                    data=report_to_pdf(report, plot_pngs=plot_pngs),
                    file_name=f"{uid}_anomaly_report.pdf",
                    mime="application/pdf",
                )

                # 5. ADDITIONAL PLOTS
                st.markdown("---")
                st.subheader("Instability Plot (2D)")
                st.altair_chart(line_chart, use_container_width=True)

                st.subheader("Model Confidence (pLDDT)")
                if plddt_df.empty:
                    st.warning("No pLDDT values found in the PDB.")
                else:
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Mean pLDDT", f"{plddt_plot_df['plddt'].mean():.1f}")
                    m_col2.metric("Min pLDDT", f"{plddt_plot_df['plddt'].min():.1f}")
                    m_col3.metric("Max pLDDT", f"{plddt_plot_df['plddt'].max():.1f}")
                    st.altair_chart(conf_chart, use_container_width=True)

                # 6. AGENT (GROQ)
                st.markdown("---")
                st.subheader("Ask the Agent")
                if st.session_state.agent_history:
                    for item in st.session_state.agent_history:
                        st.markdown(f"**You:** {item['q']}")
                        st.markdown(f"**Agent:** {item['a']}")

                with st.form("agent_form", clear_on_submit=True):
                    model_id = st.text_input("Model", value=DEFAULT_MODEL)
                    user_q = st.text_area(
                        "Question",
                        placeholder="Ask about the plots, confidence, or hotspots...",
                    )
                    submitted = st.form_submit_button("Ask Agent")

                if submitted:
                    api_key = GROQ_API_KEY or st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
                    if not api_key:
                        st.warning("Add your Groq API key in the sidebar or set GROQ_API_KEY.")
                    else:
                        with st.spinner("Thinking..."):
                            answer = run_groq_agent(
                                user_q,
                                overview_text=overview_text,
                                api_key=api_key,
                                model=model_id,
                                extra_context=st.session_state.get("agent_context", ""),
                            )
                        if answer:
                            st.session_state.agent_history.append({"q": user_q, "a": answer})
                            st.markdown(f"**You:** {user_q}")
                            st.markdown(f"**Agent:** {answer}")
                        else:
                            st.info("No response returned.")
            else: st.warning("No data found.")
else: st.info("👈 Select a disease or gene in the sidebar to begin.")
