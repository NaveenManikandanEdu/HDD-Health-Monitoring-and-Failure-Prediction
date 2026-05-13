import os
import tempfile
import matplotlib
matplotlib.use("Agg") # Prevent GUI popups
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class PDFReport(FPDF):
    def header(self):
        # Professional Header
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'HDD Fleet Health & Risk Analysis', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        # Page Numbering
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        # Gray styled section headers
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, f'  {title}', 0, 1, 'L', 1)
        self.ln(2)

    def body_text(self, text):
        # Standard text wrapper
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

def create_pdf_report(agg, model_name):
    """
    Generates a full PDF report using the dataframe 'agg' and returns it as bytes.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        sns.set_style("whitegrid")
        
        # -------------------------------------------------------------------------
        # 1. GENERATE CHARTS
        # -------------------------------------------------------------------------
        
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
                else:
                    autotext.set_color('white')
                    autotext.set_weight('bold')
            except ValueError: pass

        plt.gca().add_artist(plt.Circle((0,0), 0.70, fc='white'))
        plt.legend(wedges, bucket_counts.index, title="Health Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title("Fleet Health Overview")
        plt.tight_layout()
        chart1_path = os.path.join(temp_dir, "health_dist.png")
        plt.savefig(chart1_path)
        plt.close()

        # --- B. Bar Chart (Top Failure Mechanisms) ---
        risky = agg[agg["health_score"] < 85]
        if len(risky) > 0:
            plt.figure(figsize=(8, 4))
            top_mech = risky["dominant_mechanism"].value_counts().head(5)
            sns.barplot(x=top_mech.values, y=top_mech.index, palette="Reds_r")
            plt.title("Dominant Failure Factors (Risky Drives)")
            plt.xlabel("Count")
            plt.tight_layout()
            chart2_path = os.path.join(temp_dir, "mechanisms.png")
            plt.savefig(chart2_path)
            plt.close()
        else:
            chart2_path = None

        # --- C. Risk Distribution Histogram ---
        plt.figure(figsize=(8, 3))
        sns.histplot(agg["risk_score"], bins=50, color="#3498db", kde=True)
        plt.title("Risk Score Distribution")
        plt.xlabel("Risk Score (0-100)")
        plt.tight_layout()
        chart3_path = os.path.join(temp_dir, "risk_dist.png")
        plt.savefig(chart3_path)
        plt.close()

        # -------------------------------------------------------------------------
        # 2. COMPILE PDF
        # -------------------------------------------------------------------------
        pdf = PDFReport()
        pdf.add_page()

        pdf.section_title("Executive Summary")
        pdf.body_text(f"Total Drives Analyzed: {len(agg):,}")
        pdf.body_text(f"Model Used: {model_name}")
        pdf.body_text(f"Critical Alerts: {len(agg[agg['risk_score']>80]):,} drives exceed 80% risk.")
        
        if os.path.exists(chart1_path):
            pdf.image(chart1_path, x=20, w=170)
            pdf.ln(5)
        
        if chart2_path and os.path.exists(chart2_path):
            pdf.image(chart2_path, x=30, w=150)
            pdf.ln(5)
        
        if os.path.exists(chart3_path):
            pdf.image(chart3_path, x=30, w=150)
            pdf.ln(5)

        # --- PAGE 2: CLUSTER ANALYSIS ---
        pdf.add_page()
        pdf.section_title("Detailed Cluster Analysis")
        pdf.body_text("Drives grouped by Health Bucket and Dominant Failure Mechanism.")
        pdf.ln(5)

        buckets = ["Medium Health", "Lesser Health", "Bad Health", "Critical Health"]
        for bucket in buckets:
            b_data = agg[agg["bucket"] == bucket]
            if len(b_data) == 0: continue

            pdf.set_font('Arial', 'B', 11)
            pdf.set_text_color(255, 255, 255)
            if "Critical" in bucket: pdf.set_fill_color(139, 0, 0)
            elif "Bad" in bucket: pdf.set_fill_color(231, 76, 60)
            elif "Lesser" in bucket: pdf.set_fill_color(230, 126, 34)
            else: pdf.set_fill_color(241, 196, 15)
            
            pdf.cell(0, 8, f"{bucket.upper()} (Count: {len(b_data):,})", 0, 1, 'L', 1)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

            mechanisms = b_data["dominant_mechanism"].value_counts().head(3)
            batch_id = 1
            for mech, count in mechanisms.items():
                sub = b_data[b_data["dominant_mechanism"] == mech]
                
                # FIX: Use round instead of int to prevent 0% bugs
                pct = round((count / len(b_data)) * 100, 2)
                avg_h = int(sub["health_score"].mean())
                
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 5, f"Batch {bucket[0]}{batch_id}: {mech} Cluster", 0, 1)
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 5, f"  - Avg Score: {avg_h} | Prevalence: {pct}%", 0, 1)
                
                ids = ", ".join(sub["device_id"].head(8).astype(str).tolist())
                pdf.multi_cell(0, 5, f"  - Sample IDs: {ids}...")
                pdf.ln(3)
                batch_id += 1
            pdf.ln(3)

        # --- PAGE 3: TOP 50 TABLE ---
        pdf.add_page()
        pdf.section_title("Top 50 Most Critical Drives (Immediate Action)")
        
        pdf.set_fill_color(200, 200, 200)
        pdf.set_font('Arial', 'B', 9)
        
        pdf.cell(10, 8, '#', 1, 0, 'C', 1)
        pdf.cell(35, 8, 'Device ID', 1, 0, 'C', 1)
        pdf.cell(15, 8, 'Health', 1, 0, 'C', 1)
        pdf.cell(15, 8, 'Risk', 1, 0, 'C', 1)
        pdf.cell(0, 8, 'Primary Diagnosis', 1, 1, 'C', 1)
        
        top_50 = agg.sort_values("risk_score", ascending=False).head(50)
        pdf.set_font('Arial', '', 9)
        
        for i, row in enumerate(top_50.itertuples(), 1):
            if i % 2 == 0: pdf.set_fill_color(245, 245, 245)
            else: pdf.set_fill_color(255, 255, 255)
            
            pdf.cell(10, 6, str(i), 1, 0, 'C', 1)
            pdf.cell(35, 6, str(row.device_id), 1, 0, 'C', 1)
            pdf.cell(15, 6, f"{row.health_score:.1f}", 1, 0, 'C', 1)
            pdf.cell(15, 6, f"{row.risk_score:.1f}", 1, 0, 'C', 1)
            pdf.cell(0, 6, str(row.dominant_mechanism)[:45], 1, 1, 'L', 1)

        result = pdf.output(dest="S")
        if isinstance(result, (bytearray, bytes)):
            return bytes(result)
        return result.encode("latin-1")