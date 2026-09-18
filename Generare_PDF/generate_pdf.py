from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Table, TableStyle, Spacer


def create_fisa_tehnica():
    doc = SimpleDocTemplate(
        "Matematica.pdf",
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Stiluri personalizate
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0F172A")
    )

    math_style = ParagraphStyle(
        'MathBlock',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E1B4B"),
        backColor=colors.HexColor("#F1F5F9"),
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=1,
        borderPadding=6,
        alignment=1,
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # Antet
    story.append(Paragraph("<b>CONNECT FOUR: FEATURE-BASED Q-LEARNING</b>", title_style))
    story.append(
        Paragraph("Aproximare Liniara de Functii si Invatare prin Recompense (Reinforcement Learning)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=10))

    # 1. Problema și Tranziția
    story.append(Paragraph("1. De la Tabular Q-Learning la Aproximare Liniara", h2_style))
    text_1 = (
        "Numarul starilor in Connect Four este aproximativ <b>4.5 × 10<sup>12</sup></b>. "
        "Metoda clasica de stocare a valorilor Q intr-un dictionar esueaza din cauza lipsei de generalizare pe starile necunoscute (ex: margini), care sunt rar explorate "
        "Aproximam functia Q printr-o combinatie liniara intre un vector de ponderi <b>W</b> si un vector de trasaturi <b>Φ(s, a)</b>:"
    )
    story.append(Paragraph(text_1, body_style))
    story.append(
        Paragraph("<b>Q(s, a; W) = W<sup>T</sup> · Φ(s, a) = ∑ w<sub>i</sub> · φ<sub>i</sub>(s, a)</b>", math_style))

    # 2. Vectorul de Trăsături
    story.append(Paragraph("2. Structura Vectorului de Trasaturi Φ(s, a)", h2_style))

    data_table = [
        [Paragraph("<b>i</b>", body_style), Paragraph("<b>Trasatura (φ<sub>i</sub>)</b>", body_style),
         Paragraph("<b>Descriere Matematica </b>", body_style)],
        ["1", "Bias", "Constanta = 1.0 (Ajustare nivel de baza)"],
        ["2", "Control Centru", "Scor proportional cu proximitatea de coloana centrala, 3"],
        ["3", "Linii 2 (A)", "Numarul de ferestre de 4 celule cu 2 piese proprii si 2 libere"],
        ["4", "Linii 3 (A)", "Numarul de amenințări proprii de 3 piese în linie"],
        ["5", "Victorie (A)", "Numarul de linii complete de 4 piese proprii"],
        ["6", "Blocare 3 (B)", "Numarul de amenintari de 3 piese ale adversarului blocate"],
        ["7", "Blocare 4 (B)", "Blocarea directa a unei linii de 3 a adversarului"]
    ]

    t = Table(data_table, colWidths=[25, 110, 385])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # 3. Regula de Învațare SGD & TD-Error
    story.append(Paragraph("3. Optimizarile Ponderilor prin Gradient Descent (SGD)", h2_style))
    text_3 = (
        "La fiecare pas, agentul evalueaza diferenta dintre estimarea curenta si recompensa reala plus valoarea starii viitoare "
        "(<b>Eroarea Temporal Difference - TD Error</b>):"
    )
    story.append(Paragraph(text_3, body_style))
    story.append(Paragraph(
        "<b>δ<sub>t</sub> = R<sub>t+1</sub> + γ · max<sub>a'</sub> Q(s<sub>t+1</sub>, a'; W) - Q(s<sub>t</sub>, a<sub>t</sub>; W)</b>",
        math_style))

    text_update = "Ponderile <b>W</b> sunt actualizate prin ajustarea Stochastic Gradient Descent:"
    story.append(Paragraph(text_update, body_style))
    story.append(Paragraph("<b>W = W + α · δ<sub>t</sub> · Φ(s<sub>t</sub>, a<sub>t</sub>)</b>", math_style))

    # 4. Parametrii de Antrenament
    story.append(Paragraph("4. Politica de Selectie (ε-Greedy) si Parametri", h2_style))
    text_4 = (
        "<b>• Invatare (α = 0.005, γ = 0.9):</b> Balanta optima intre recompensele imediate si strategiile pe termen lung.<br/>"
        "<b>• Explorare (ε-Greedy):</b> In antrenament ε scade de la 0.3 la 0.01. În meciul live cu omul <b>ε = 0.0</b> (exploatare 100% deterministă)."
    )
    story.append(Paragraph(text_4, body_style))

    doc.build(story)


if __name__ == "__main__":
    create_fisa_tehnica()