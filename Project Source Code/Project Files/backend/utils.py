import base64
import datetime
from fpdf import FPDF
from pathlib import Path
import cv2

def generate_medical_summary(prediction, confidence):
    if "Glioma" in prediction:
        return (f"AI Analysis indicates the PRESENCE of a Glioma with a confidence of {confidence*100:.2f}%. "
                f"The segmented region (shown in the structural mask) isolates the main tumor mass. "
                f"Grad-CAM++ analysis highlights the key structural anomalies contributing to this decision. "
                f"Immediate clinical review and follow-up MRI sequences are recommended.")
    else:
        return (f"AI Analysis indicates NO TUMOR detected with a confidence of {confidence*100:.2f}%. "
                f"The structural presentation appears normal within the analyzed MRI slices. "
                f"Standard routine follow-up as clinically indicated.")

def encode_image_to_base64(image_arr):
    _, buffer = cv2.imencode('.png', image_arr)
    return base64.b64encode(buffer).decode('utf-8')

class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Automated Glioma Detection Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(original_img_path, prediction, confidence, summary, out_path="outputs/reports"):
    Path(out_path).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"{out_path}/report_{timestamp}.pdf"
    
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Prediction Details", 0, 1)
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 10, f"Diagnosis: {prediction}", 0, 1)
    pdf.cell(0, 10, f"Confidence: {confidence*100:.2f}%", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Clinical Summary", 0, 1)
    pdf.set_font("helvetica", "", 12)
    pdf.multi_cell(0, 10, summary)
    pdf.ln(10)
    
    # Can optionally embed the image into the PDF here
    # pdf.image(original_img_path, w=100)
    
    pdf.output(pdf_path)
    return pdf_path
