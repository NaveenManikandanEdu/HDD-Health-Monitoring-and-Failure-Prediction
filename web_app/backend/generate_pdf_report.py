import os
import shutil
import tempfile
import matplotlib
# Force matplotlib to not use any Xwindow backend
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'HDD Fleet Health & Risk Analysis', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, f'  {title}', 0, 1, 'L', 1)
        self.ln(2)

    def body_text(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

def create_pdf_report(agg, model_name="Enterprise_V1"):
    temp_dir = tempfile.mkdtemp()
    
    try:
        sns.set_style("whitegrid")
        
        # --- A. Donut Chart ---
        bucket_counts = agg["bucket"].value_counts()
        colors = {"Very Good Health":"#2ecc71", "Medium Health":"#f1c40f", 
                  "Lesser Health":"#e67e22", "Bad Health":"#e74c3c", "Critical Health":"#8b0000"}
        chart_colors = [colors.get(x, "#333") for x in bucket_counts.index]
        
        plt.figure(figsize=(10, 5))
        wedges, texts, autotexts = plt.pie(
            bucket_counts, labels=None, autopct='%1.1f%%', 
            colors=chart_colors, pctdistance=0.85, startangle=90
        )
        for autotext in autotexts:
            try:
                val = float(autotext.get_text().strip('%'))
                if val < 2.0: autotext.set_text('')
                else: autotext.set_color('white'); autotext.set_weight('bold')
            except: pass

        plt.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
        plt.legend(wedges, bucket_counts.index, title="Health Categories", 
                   loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title("Fleet Health Overview")
        plt.tight_layout()
        chart1_path = os.path.join(temp_dir, "health_dist.png")
        plt.savefig(chart1_path)
        plt.close()

        # --- B. Risk Distribution ---
        plt.figure(figsize=(8, 3))
        sns.histplot(agg['risk_score'], bins=30, color="#3498db", kde=True)
        plt.title("Risk Score Probability Distribution")
        plt.tight_layout()
        chart2_path = os.path.join(temp_dir, "risk_dist.png")
        plt.savefig(chart2_path)
        plt.close()

        # --- PDF COMPILE ---
        pdf = PDFReport()
        pdf.add_page()

        pdf.section_title("Executive Summary")
        pdf.body_text(f"Total Assets Analyzed: {len(agg):,}")
        pdf.body_text(f"Method: Pattern Similarity Analysis (Z-Calibrated)")
        pdf.body_text(f"Urgent Alerts: {len(agg[agg['risk_score'] > 80])} critical risks found.")
        
        pdf.image(chart1_path, x=20, y=60, w=170)
        pdf.image(chart2_path, x=30, y=140, w=150)

        pdf.add_page()
        pdf.section_title("Detailed Cluster Analysis")
        sort_order = ["Critical Health", "Bad Health", "Lesser Health", "Medium Health"]
        for bucket in sort_order:
            b_data = agg[agg["bucket"] == bucket]
            if b_data.empty: continue

            pdf.set_font('Arial', 'B', 11)
            pdf.set_text_color(255, 255, 255)
            if "Critical" in bucket: pdf.set_fill_color(139, 0, 0)
            elif "Bad" in bucket: pdf.set_fill_color(231, 76, 60)
            else: pdf.set_fill_color(230, 126, 34)
            
            pdf.cell(0, 8, f"  {bucket.upper()} (Count: {len(b_data):,})", 0, 1, 'L', 1)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

            mechanisms = b_data["dominant_mechanism"].value_counts().head(3)
            batch_id = 1
            for mech, count in mechanisms.items():
                sub = b_data[b_data["dominant_mechanism"] == mech]
                pct = round((count / len(b_data)) * 100, 1)
                avg_h = int(sub["health_score"].mean()) if not sub.empty else 0
                
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 5, f"Batch {bucket[0]}{batch_id}: {mech} Cluster", 0, 1)
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 5, f"   - Avg Health: {avg_h}% | Prevalence: {pct}%", 0, 1)
                pdf.multi_cell(0, 5, f"   - Samples: {', '.join(sub['device_id'].head(6).astype(str).tolist())}...")
                pdf.ln(2)
                batch_id += 1
            pdf.ln(2)

        pdf.add_page()
        pdf.section_title("Top 50 Risk Inventory")
        top_50 = agg.sort_values("risk_score", ascending=False).head(50)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(10, 8, '#', 1); pdf.cell(45, 8, 'Device ID', 1); pdf.cell(20, 8, 'Health %', 1); pdf.cell(20, 8, 'Risk %', 1); pdf.cell(0, 8, 'Diagnosis', 1, 1)
        
        pdf.set_font('Arial', '', 9)
        for i, row in enumerate(top_50.itertuples(), 1):
            pdf.cell(10, 6, str(i), 1); pdf.cell(45, 6, str(row.device_id), 1); pdf.cell(20, 6, str(row.health_score), 1); pdf.cell(20, 6, str(row.risk_score), 1); pdf.cell(0, 6, str(row.dominant_mechanism)[:35], 1, 1)

        # CRITICAL FIX: Ensure output is returned as standard bytes
        result = pdf.output()
        if isinstance(result, (bytearray, bytes)):
            return bytes(result)
        return result.encode('latin-1')

    finally:
        shutil.rmtree(temp_dir)