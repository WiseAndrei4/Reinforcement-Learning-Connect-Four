from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def create_fisa_copii():
    doc = SimpleDocTemplate(
        "Explicatie_Amuzanta.pdf",
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "KidsTitle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#DC2626"),
        alignment=1,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "KidsSubTitle",
        parent=styles["Normal"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1D4ED8"),
        alignment=1,
        spaceAfter=12,
    )

    h2_style = ParagraphStyle(
        "KidsH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "KidsBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
    )

    callout_style = ParagraphStyle(
        "CalloutBox",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#854D0E"),
        backColor=colors.HexColor("#FEF08A"),
        borderColor=colors.HexColor("#EAB308"),
        borderWidth=1.5,
        borderPadding=8,
        alignment=1,
        spaceBefore=6,
        spaceAfter=8,
    )

    story = []

    # Titlu & Introducere
    story.append(
        Paragraph("🤖 CUM GANDESTE ROBOTUL CONNECT FOUR?", title_style)
    )
    story.append(
        Paragraph(
            "Afla secretul din spatele inteligentei artificiale de la stand!",
            subtitle_style,
        )
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#DC2626"),
            spaceAfter=12,
        )
    )

    # Sectiunea 1
    story.append(
        Paragraph("🧠 Memorare vs. Instinct (Superputerea AI-ului)", h2_style)
    )
    story.append(
        Paragraph(
            "Daca un robot ar incerca sa invete pe de rost toate mutarile posibile din joc, "
            "ar trebui sa citeasca o biblioteca uriasa cu <b>4.5 Trilioane de carti</b>! "
            "Robotul nostru nu memoreaza tabla, ci foloseste <b>instinctul</b> — exact ca un jucator uman experimentat.",
            body_style,
        )
    )

    # Sectiunea 2
    story.append(Paragraph("🔍 Cele 3 Superputeri ale Robotului:", h2_style))

    # Fiecare element este îmbrăcat în Paragraph pentru wrap automat
    data_table = [
        [
            Paragraph("<b>Superputere</b>", body_style),
            Paragraph("<b>Ce face Robotul?</b>", body_style),
        ],
        [
            Paragraph("Scutul Defensiv", body_style),
            Paragraph(
                "Cand vede ca ai pus 3 piese in linie, sare imediat si te"
                " blocheaza!",
                body_style,
            ),
        ],
        [
            Paragraph("Atacul Iute", body_style),
            Paragraph(
                "Cauta sa uneasca propriile piese pentru a face 4 in linie si a"
                " castiga.",
                body_style,
            ),
        ],
        [
            Paragraph("Controlul Centrului", body_style),
            Paragraph(
                "Aseaza piese pe coloana din mijloc pentru ca de acolo are cele mai"
                " multe sanse.",
                body_style,
            ),
        ],
    ]

    t = Table(data_table, colWidths=[130, 335])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#93C5FD")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(t)

    # Sectiunea 3
    story.append(Paragraph("🎮 Cum Invata? (Ca intr-un Joc Video)", h2_style))
    story.append(
        Paragraph(
            "Robotul s-a antrenat jucand <b>zeci de mii de meciuri impotriva lui insusi</b> in doar cateva secunde! "
            "Dupa fiecare mutare, el primeste puncte:<br/>"
            "• <b>+10 Puncte (Recompensa):</b> Cand castiga un meci.<br/>"
            "• <b>-10 Puncte (Pedeapsa):</b> Cand pierde sau face o greseala.<br/>"
            "Astfel, isi ajusteaza 'punctele de experienta' pana devine imbatabil!",
            body_style,
        )
    )

    # Provocare
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>🔥 PROVOCARE PENTRU TINE LA STAND:</b><br/>"
            "Incearca sa-l pacalesti atacand pe margini sau creand o <b>dubla amenintare</b> (doua linii de 3 simultan)! Poti sa-l bati?",
            callout_style,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    create_fisa_copii()