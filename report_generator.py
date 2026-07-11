import os
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(prediction_record, output_pdf_path):
    """
    Generates a beautifully formatted PDF report for a prediction record.
    
    Args:
        prediction_record (dict): Row dict from the predictions database containing:
            id, image_name, predicted_class, confidence, probabilities, date, time, 
            processing_time, original_image_path, gradcam_image_path
        output_pdf_path (str): Filepath to save the generated PDF.
    """
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#0f172a'), # Slate 900
        spaceAfter=15,
        alignment=1 # Centered
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#475569'), # Slate 600
        spaceAfter=25,
        alignment=1 # Centered
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1e3a8a'), # Dark Blue
        spaceBefore=15,
        spaceAfter=10,
        borderPadding=2
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'), # Slate 700
        leading=14
    )
    
    body_bold_style = ParagraphStyle(
        'BodyBoldTextCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#94a3b8'), # Slate 400
        leading=12,
        alignment=1 # Centered
    )
    
    story = []
    
    # 1. Header Section
    story.append(Paragraph("BRAIN MRI SCAN ANALYSIS REPORT", title_style))
    report_date = datetime.strptime(prediction_record['date'], "%Y-%m-%d").strftime("%B %d, %Y")
    report_time = prediction_record['time']
    story.append(Paragraph(f"Report ID: BTR-{prediction_record['id']:06d}   |   Date Generated: {report_date} {report_time}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # 2. Metadata Section (Patient / Scan Details)
    story.append(Paragraph("Analysis Metadata", section_title_style))
    
    # Build data table
    class_name = prediction_record['predicted_class']
    confidence_pct = f"{prediction_record['confidence'] * 100:.2f}%"
    proc_time_sec = f"{prediction_record['processing_time']:.4f} seconds"
    
    # Sort probabilities descending
    sorted_probs = sorted(prediction_record['probabilities'].items(), key=lambda x: x[1], reverse=True)
    probs_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in sorted_probs])
    
    metadata_data = [
        [Paragraph("<b>File Name:</b>", body_style), Paragraph(prediction_record['image_name'], body_style),
         Paragraph("<b>Analysis Time:</b>", body_style), Paragraph(proc_time_sec, body_style)],
        [Paragraph("<b>Predicted Class:</b>", body_style), Paragraph(f"<b>{class_name}</b>", body_bold_style),
         Paragraph("<b>Confidence Score:</b>", body_style), Paragraph(f"<b>{confidence_pct}</b>", body_bold_style)],
        [Paragraph("<b>Class Distribution:</b>", body_style), Paragraph(probs_str, body_style), "", ""]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
    metadata_table.setStyle(TableStyle([
        ('SPAN', (1, 2), (3, 2)), # Span class distribution across remaining columns
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 20))
    
    # 3. Image Section (Side-by-Side original MRI and Grad-CAM)
    story.append(Paragraph("Visualizations", section_title_style))
    
    orig_path = prediction_record['original_image_path']
    gradcam_path = prediction_record['gradcam_image_path']
    
    # Fallback to absolute paths if not absolute already
    project_dir = Path(__file__).resolve().parent
    if not os.path.isabs(orig_path):
        orig_path = str((project_dir / orig_path).resolve())
    if not os.path.isabs(gradcam_path):
        gradcam_path = str((project_dir / gradcam_path).resolve())
        
    image_table_data = []
    
    # Add images if they exist
    if os.path.exists(orig_path) and os.path.exists(gradcam_path):
        # Resize images to 3.2 inches wide (roughly 230 points) to fit side-by-side
        img_w, img_h = 3.2*inch, 3.2*inch
        orig_img = Image(orig_path, width=img_w, height=img_h)
        gradcam_img = Image(gradcam_path, width=img_w, height=img_h)
        
        image_table_data = [
            [orig_img, gradcam_img],
            [Paragraph("<b>Original MRI Scan</b>", body_style), Paragraph("<b>Grad-CAM Explainable AI Heatmap</b>", body_style)]
        ]
    else:
        # Fallback text if images not found
        image_table_data = [
            [Paragraph("<i>Original MRI Image File missing</i>", body_style), 
             Paragraph("<i>Grad-CAM Image File missing</i>", body_style)]
        ]
        
    image_table = Table(image_table_data, colWidths=[3.5*inch, 3.5*inch])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(image_table)
    story.append(Spacer(1, 20))
    
    # 4. Findings & Educational context
    story.append(Paragraph("Clinical Explanation (Educational)", section_title_style))
    
    explanations = {
        "Glioma": "Gliomas are primary brain tumors that originate from glial cells in the brain or spinal cord. The Grad-CAM heatmap highlights the tumor boundaries and surrounding edema (swelling) which are crucial for surgical planning and radiation therapy.",
        "Meningioma": "Meningiomas are typically benign, slow-growing tumors arising from the meninges (the membranes covering the brain and spinal cord). On MRI, they appear as well-circumscribed, extra-axial masses. The Grad-CAM highlights the dural attachment area.",
        "Pituitary Tumor": "Pituitary tumors develop in the pituitary gland at the base of the brain. They can affect hormone production and press on the optic chiasm, causing visual fields defects. The heatmap focuses on the sella turcica area.",
        "No Tumor": "The classification model did not find structural abnormalities consistent with glioma, meningioma, or pituitary tumors in the uploaded MRI scan. The Grad-CAM displays background attention or general tissue structures."
    }
    
    explanation_text = explanations.get(class_name, "Brain tumor scans are analyzed using deep learning networks. Grad-CAM shows the activation maps pointing to regions contributing most to the neural network's final output classification.")
    story.append(Paragraph(explanation_text, body_style))
    story.append(Spacer(1, 30))
    
    # 5. Footer / Disclaimer section (Keep together at the bottom)
    disclaimer_elements = [
        Spacer(1, 10),
        Paragraph("<b>EDUCATIONAL DISCLAIMER:</b> This report is generated by a computer vision machine learning model trained on a research dataset. It is provided strictly for educational, research, and portfolio demonstration purposes. It does NOT constitute medical advice, diagnostic services, or professional consultation. If this scan belongs to a patient, please seek immediate review by a licensed healthcare professional, board-certified radiologist, or medical doctor.", disclaimer_style)
    ]
    story.append(KeepTogether(disclaimer_elements))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report generated at {output_pdf_path}")
