#!/usr/bin/env python3
# Generate LaTeX tables for paper from ablation results

import os
import re
import json
from typing import Dict, List, Optional
from pathlib import Path


class PaperTableGenerator:
    def __init__(self, output_base: str = 'outputs'):
        self.output_base = output_base
        
    def parse_eval_log(self, log_path: str) -> Optional[Dict[str, float]]:
        " \Parse evaluation log to extract metrics\\ if not os.path.exists(log_path): return None metrics = {} with open(log_path, r) as f: content = f.read() # Parse HOTA, MOTA, IDF1, etc. patterns =  HOTA: rHOTA\s*[:=]\s*([\d.]+), MOTA: rMOTA\s*[:=]\s*([\d.]+), IDF1: rIDF1\s*[:=]\s*([\d.]+), AssA: rAssA\s*[:=]\s*([\d.]+), DetA: rDetA\s*[:=]\s*([\d.]+), IDSW: rIDSW\s*[:=]\s*(\d+), } for name, pattern in patterns.items(): match = re.search(pattern, content, re.IGNORECASE) if match: metrics[name] = float(match.group(1)) return metrics if metrics else None def generate_ablation_table(self, experiments: Dict[str, str]) -> str: \Generate LaTeX table for ablation study\\ latex = [] latex.append(r\begin{table}[t]) latex.append(r\centering) latex.append(r\caption{Ablation study on MOT17 validation set.}) latex.append(r\label{tab:ablation}) latex.append(r\begin{tabular}{l|ccccc}) latex.append(r\toprule) latex.append(rMethod & HOTA & MOTA & IDF1 & AssA & DetA \\) latex.append(r\midrule) for name, exp_dir in experiments.items(): log_path = os.path.join(self.output_base, exp_dir, eval.log) metrics = self.parse_eval_log(log_path) if metrics: row = f{name} & {metrics.get(\HOTA\, \-\):.1f} &  row += f{metrics.get(\MOTA\, \-\):.1f} &  row += f{metrics.get(\IDF1\, \-\):.1f} &  row += f{metrics.get(\AssA\, \-\):.1f} &  row += f{metrics.get(\DetA\, \-\):.1f} \\\\ else: row = f{name} & - & - & - & - & - \\\\ latex.append(row) latex.append(r\bottomrule) latex.append(r\end{tabular}) latex.append(r\end{table}) return \n.join(latex) def generate_num_bands_table(self) -> str: \Generate table for number of bands ablation\\ experiments =  K=2: ablation_num_bands_K2, K=4 (Ours): ablation_num_bands_K4, K=6: ablation_num_bands_K6, K=8: ablation_num_bands_K8, } return self.generate_ablation_table(experiments) def generate_component_table(self) -> str: \Generate table for component ablation\\ experiments =  Baseline (w/o FA): ablation_no_freq_aware, w/o Dual-Branch: ablation_no_dual_branch, w/o Mamba: ablation_no_mamba, w/o Ortho Loss: ablation_no_ortho, FM-Track (Full): bytetrack_fa_mot_mot17_v4_topconf, } return self.generate_ablation_table(experiments) def main(): generator = PaperTableGenerator(outputs) print(=== Ablation Table: Number of Bands ===) print(generator.generate_num_bands_table()) print() print(=== Ablation Table: Components ===) print(generator.generate_component_table()) if __name__ == __main__: main() ENDOFFILE echo Created generate_paper_tables.py
