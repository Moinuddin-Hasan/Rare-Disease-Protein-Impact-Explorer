import pandas as pd
import numpy as np
import requests
import io
import streamlit as st
import xml.etree.ElementTree as ET
from Bio.PDB import PDBParser, Polypeptide
from sklearn.cluster import DBSCAN
import os
import re

# DATA: HYDRO_MAP
# Purpose: Static lookup table for the Kyte-Doolittle Hydrophobicity scale of amino acids.
HYDRO_MAP = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8,
    'GLY': -0.4, 'THR': -0.7, 'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6,
    'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

# FUNCTION: load_rare_disease_catalog
# Purpose: Parses the Orphanet XML file to create a searchable map of rare diseases and their primary associated genes.
# Input: xml_path (String) - Path to the Orphanet 'en_product6.xml' file.
# Output: (disease_to_gene, gene_to_disease) (Tuple of Dictionaries) - Two maps for bi-directional search synchronization.
@st.cache_data
def load_rare_disease_catalog(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        d_to_g = {}
        g_to_d = {}

        for disorder in root.findall(".//Disorder"):
            name_element = disorder.find("Name")
            gene_elements = disorder.findall(".//Gene")
            
            if name_element is not None and gene_elements:
                disease_name = name_element.text
                primary_gene = gene_elements[0].find("Symbol").text
                d_to_g[disease_name] = primary_gene
                g_to_d[primary_gene] = disease_name
                
        return d_to_g, g_to_d
    except Exception as e:
        print(f"Orphanet XML Error: {e}")
        return {}, {}

# FUNCTION: get_uniprot_id
# Purpose: Connects to the UniProt REST API to translate a human gene symbol into a unique Protein Accession ID.
# Input: gene_symbol (String).
# Output: uniprot_id (String/None).
@st.cache_data
def get_uniprot_id(gene_symbol):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_symbol} AND organism_id:9606&format=json"
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        if data.get('results'):
            return data['results'][0]['primaryAccession']
        return None
    except Exception:
        return None

# FUNCTION: resolve_uniprot_id
# Purpose: Resolve a gene symbol or UniProt accession to a UniProt ID. If input looks like a UniProt ID, return as-is.
# Input: gene_or_uniprot (String).
# Output: uniprot_id (String/None).
def resolve_uniprot_id(gene_or_uniprot):
    if not gene_or_uniprot:
        return None

    token = gene_or_uniprot.strip().upper()
    uniprot_pattern = r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{2}[A-Z0-9]{2}[0-9])$"
    if re.match(uniprot_pattern, token):
        return token

    return get_uniprot_id(token)

# FUNCTION: get_af_structure_url
# Purpose: Uses the AlphaFold API to find the REAL current download link for a protein.
# Input: uniprot_id (String).
# Output: pdb_url (String/None).
def get_af_structure_url(uniprot_id):
    if not uniprot_id:
        return None
    
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id.strip()}"
    
    try:
        response = requests.get(api_url, verify=False, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get('pdbUrl')
    except Exception as e:
        print(f"AlphaFold API Lookup Error: {e}")
    
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id.strip()}-F1-model_v4.pdb"

# FUNCTION: get_protein_anomalies_smart
# Purpose: Scans the AlphaMissense dataset using chunked reading to extract pathogenicity scores and residue positions.
# Input: uniprot_id (String), file_path (String).
# Output: protein_df (DataFrame).
@st.cache_data
def get_protein_anomalies_smart(uniprot_id, file_path):
    col_map = {5: 'uniprot_id', 7: 'variant', 8: 'am_pathogenicity', 9: 'am_class'}
    
    try:
        chunks = pd.read_csv(
            file_path, 
            sep='\t', 
            compression='gzip', 
            comment='#', 
            usecols=list(col_map.keys()),
            names=[col_map[i] for i in sorted(col_map.keys())],
            header=None,
            chunksize=200000,
            engine='c'
        )
        
        relevant_chunks = []
        for chunk in chunks:
            chunk.columns = ['uniprot_id', 'variant', 'am_pathogenicity', 'am_class']
            chunk['uniprot_id'] = chunk['uniprot_id'].astype(str).str.strip()
            filtered = chunk[chunk['uniprot_id'] == uniprot_id].copy()
            
            if not filtered.empty:
                filtered['residue_num'] = filtered['variant'].str.extract('(\d+)').fillna(0).astype(int)
                relevant_chunks.append(filtered)
                
        if not relevant_chunks:
            return pd.DataFrame(columns=['uniprot_id', 'variant', 'am_pathogenicity', 'am_class', 'residue_num'])
            
        return pd.concat(relevant_chunks)
        
    except Exception as e:
        st.error(f"Error reading compressed file: {e}")
        return pd.DataFrame()

# FUNCTION: get_protein_anomalies_cached
# Purpose: Disk cache for per-protein AlphaMissense slices to avoid re-scanning the large TSV.
# Input: uniprot_id (String), file_path (String), cache_dir (String).
# Output: protein_df (DataFrame).
def get_protein_anomalies_cached(uniprot_id, file_path, cache_dir="data/cache"):
    if not uniprot_id:
        return pd.DataFrame()

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{uniprot_id}.csv")

    try:
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
    except Exception:
        pass

    df = get_protein_anomalies_smart(uniprot_id, file_path)
    if not df.empty:
        try:
            df.to_csv(cache_path, index=False)
        except Exception:
            pass
    return df

# FUNCTION: detect_structural_hotspots
# Purpose: Sophisticated AI logic that uses DBSCAN to find 3D clusters and attaches pathogenicity scores for impact ranking.
# Input: local_pdb_path (String), high_risk_df (DataFrame), eps (Float), min_samples (Int).
# Output: cluster_results (DataFrame).
def detect_structural_hotspots(local_pdb_path, high_risk_df, eps=10.0, min_samples=3):
    cols = ['residue_num', 'cluster_id', 'score']
    if high_risk_df.empty or not local_pdb_path or not os.path.exists(local_pdb_path):
        return pd.DataFrame(columns=cols)

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", local_pdb_path)
        model = structure[0]
        
        coords, res_ids, scores = [], [], []
        target_residues = high_risk_df.drop_duplicates('residue_num')
        
        for _, row in target_residues.iterrows():
            res_num = int(row['residue_num'])
            try:
                residue = model['A'][res_num]
                atom = residue['CA']
                coords.append(atom.get_coord())
                res_ids.append(res_num)
                scores.append(row['am_pathogenicity'])
            except KeyError:
                continue
                
        if len(coords) < min_samples:
            return pd.DataFrame(columns=cols)

        x_data = np.array(coords)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(x_data)
        
        return pd.DataFrame({
            'residue_num': res_ids,
            'cluster_id': dbscan.labels_,
            'score': scores
        })
    except Exception:
        return pd.DataFrame(columns=cols)

# FUNCTION: download_pdb_locally
# Purpose: Downloads the PDB file from AlphaFold and saves it to a local temporary folder.
# Input: url (String), uniprot_id (String).
# Output: local_path (String/None).
def download_pdb_locally(url, uniprot_id):
    if not url:
        return None
        
    try:
        os.makedirs("temp_pdb", exist_ok=True)
        local_path = os.path.join("temp_pdb", f"{uniprot_id.strip()}.pdb")
        
        response = requests.get(url, verify=False, timeout=15) 
        
        if response.status_code == 200:
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return local_path
        else:
            return None
    except Exception as e:
        print(f"CRITICAL ERROR during download: {e}")
        return None

# FUNCTION: extract_plddt_from_pdb
# Purpose: Extract per-residue pLDDT from the B-factor field of an AlphaFold PDB.
# Input: local_pdb_path (String).
# Output: plddt_df (DataFrame).
def extract_plddt_from_pdb(local_pdb_path):
    if not local_pdb_path or not os.path.exists(local_pdb_path):
        return pd.DataFrame(columns=["residue_num", "plddt"])

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", local_pdb_path)
        model = structure[0]
        chain = model["A"] if "A" in model else next(model.get_chains())

        rows = []
        for residue in chain.get_residues():
            if "CA" not in residue:
                continue
            res_id = residue.get_id()
            if res_id[0] != " ":
                continue
            res_num = int(res_id[1])
            b_factor = float(residue["CA"].get_bfactor())
            rows.append({"residue_num": res_num, "plddt": b_factor})

        df = pd.DataFrame(rows)
        return df.sort_values("residue_num").drop_duplicates("residue_num")
    except Exception:
        return pd.DataFrame(columns=["residue_num", "plddt"])

# FUNCTION: get_sequence_from_pdb
# Purpose: Extracts the 1-letter amino acid sequence from the PDB file for the interactive ribbon.
# Input: local_pdb_path (String).
# Output: sequence (String).
def get_sequence_from_pdb(local_pdb_path):
    if not local_pdb_path or not os.path.exists(local_pdb_path):
        return ""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", local_pdb_path)
        ppb = Polypeptide.PPBuilder()
        sequence = ""
        for pp in ppb.build_peptides(structure):
            sequence += str(pp.get_sequence())
        return sequence
    except:
        return ""

# FUNCTION: apply_pdb_theme
# Purpose: Rewrites the PDB B-factors based on a selected theme (Hydrophobicity, AI Risk, or Confidence).
# Input: original_pdb (String), theme_type (String), protein_df (DataFrame).
# Output: themed_path (String/None).
def apply_pdb_theme(original_pdb, theme_type, protein_df=None):
    if not original_pdb or not os.path.exists(original_pdb):
        return None
    
    themed_path = original_pdb.replace(".pdb", f"_{theme_type}.pdb")
    patho_map = {}
    if theme_type == "Anomaly" and protein_df is not None:
        patho_map = protein_df.set_index('residue_num')['am_pathogenicity'].to_dict()

    with open(original_pdb, 'r') as f_in, open(themed_path, 'w') as f_out:
        for line in f_in:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                
                if theme_type == "Hydrophobicity":
                    val = HYDRO_MAP.get(res_name, 0.0)
                elif theme_type == "Anomaly":
                    val = patho_map.get(res_num, 0.0) * 100 
                else:
                    f_out.write(line)
                    continue
                
                new_line = line[:60] + f"{val:6.2f}" + line[66:]
                f_out.write(new_line)
            else:
                f_out.write(line)
    return themed_path