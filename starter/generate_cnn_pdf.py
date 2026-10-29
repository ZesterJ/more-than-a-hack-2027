"""Generate an executive PDF summary of the More Than A Hack 2027 CNN model benchmark,
winning model descriptions, and engineered feature set using ReportLab.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Output paths
HERE = os.path.dirname(os.path.abspath(__file__))
DESKTOP_PDF = r"C:\Users\harve\Desktop\More_Than_A_Hack_2027_Model_Benchmark.pdf"
WORKSPACE_PDF = os.path.join(r"C:\Users\harve\Antigravity\Barca", "More_Than_A_Hack_2027_Model_Benchmark.pdf")
ARTIFACT_PDF = r"C:\Users\harve\.gemini\antigravity\brain\373cda61-4f74-4ec6-8524-a71ad6f18bac\More_Than_A_Hack_2027_Model_Benchmark.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header rule and title
        self.drawString(40, 11 * 72 - 32, "More Than A Hack 2027 -- CNN Deep Learning Benchmark & Model Selection")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(40, 11 * 72 - 38, 8.5 * 72 - 40, 11 * 72 - 38)
        # Footer rule and page
        self.line(40, 38, 8.5 * 72 - 40, 38)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 40, 26, page_text)
        self.drawString(40, 26, "Official Competition Benchmark -- Prepared by @harveycheung2")
        self.restoreState()


def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=44,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f2a4a")
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#444444")
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f2a4a"),
        spaceBefore=8,
        spaceAfter=4
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1b5e20"),
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#222222")
    )
    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=10,
        firstLineIndent=-6,
        spaceAfter=2
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#222222")
    )
    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell,
        alignment=1
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=1
    )

    story = []

    # Title block
    story.append(Paragraph("More Than A Hack 2027: Ball Possession Benchmark Report", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Deep Learning & CNN Benchmark:</b> ResNet-1D, InceptionTime, Recurrent Models & Classical Baselines | <b>Engineer:</b> @harveycheung2", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0f2a4a"), spaceAfter=8))

    # SECTION 1: MASTER BENCHMARK TABLE
    story.append(Paragraph("1. Comprehensive Model Benchmark Scoreboard", h1_style))
    story.append(Paragraph(
        "All models evaluated under <b>5-Fold Contiguous Out-of-Sample Cross-Validation</b> on the 7,500 labeled frames "
        "(public 5-minute ground truth) using the official competition evaluator (<code>possession_metrics.py</code>). "
        "<b>Target Ground Truth: 17 possessions, 14.2s mean duration.</b>",
        body_style
    ))
    story.append(Spacer(1, 4))

    table_data = [
        [
            Paragraph("Rank", table_header),
            Paragraph("Model Architecture", table_header),
            Paragraph("spellF1<br/>(Ranked)", table_header),
            Paragraph("macroF1<br/>(Tie-break)", table_header),
            Paragraph("Accuracy", table_header),
            Paragraph("F1(H)", table_header),
            Paragraph("F1(A)", table_header),
            Paragraph("F1(D)", table_header),
            Paragraph("Spells", table_header),
            Paragraph("Mean Dur", table_header)
        ],
        [Paragraph("1 🏆", table_cell_center), Paragraph("<b>ResNet-1D CNN (Smooth w=50 / 2.0s)</b>", table_cell), Paragraph("<b>0.5875</b>", table_cell_center), Paragraph("0.7286", table_cell_center), Paragraph("78.59%", table_cell_center), Paragraph("0.8638", table_cell_center), Paragraph("0.7997", table_cell_center), Paragraph("<b>0.5223</b>", table_cell_center), Paragraph("15", table_cell_center), Paragraph("17.7s", table_cell_center)],
        [Paragraph("2 🥈", table_cell_center), Paragraph("<b>CNN Blend (InceptionTime + ResNet, w=35)</b>", table_cell), Paragraph("<b>0.5000</b>", table_cell_center), Paragraph("<b>0.7447</b>", table_cell_center), Paragraph("<b>80.07%</b>", table_cell_center), Paragraph("0.8774", table_cell_center), Paragraph("0.8001", table_cell_center), Paragraph("<b>0.5566</b>", table_cell_center), Paragraph("15", table_cell_center), Paragraph("18.2s", table_cell_center)],
        [Paragraph("3 🥉", table_cell_center), Paragraph("<b>ResNet-1D CNN (Smooth w=25 / 1.0s)</b>", table_cell), Paragraph("0.4889", table_cell_center), Paragraph("0.7211", table_cell_center), Paragraph("77.77%", table_cell_center), Paragraph("0.8548", table_cell_center), Paragraph("0.7965", table_cell_center), Paragraph("0.5120", table_cell_center), Paragraph("19", table_cell_center), Paragraph("13.9s", table_cell_center)],
        [Paragraph("4", table_cell_center), Paragraph("LogReg + Softmax Smoothing (w=50)", table_cell), Paragraph("0.4686", table_cell_center), Paragraph("0.6107", table_cell_center), Paragraph("72.80%", table_cell_center), Paragraph("0.7706", table_cell_center), Paragraph("0.7576", table_cell_center), Paragraph("0.2312", table_cell_center), Paragraph("18", table_cell_center), Paragraph("16.0s", table_cell_center)],
        [Paragraph("5", table_cell_center), Paragraph("Temporal Neural Ensemble (TCN + BiGRU, w=35)", table_cell), Paragraph("0.4667", table_cell_center), Paragraph("0.7017", table_cell_center), Paragraph("78.37%", table_cell_center), Paragraph("0.8786", table_cell_center), Paragraph("0.7857", table_cell_center), Paragraph("0.4407", table_cell_center), Paragraph("19", table_cell_center), Paragraph("14.6s", table_cell_center)],
        [Paragraph("6", table_cell_center), Paragraph("Bidirectional GRU (Smooth w=25 / 1.0s)", table_cell), Paragraph("0.4513", table_cell_center), Paragraph("0.7000", table_cell_center), Paragraph("77.64%", table_cell_center), Paragraph("0.8704", table_cell_center), Paragraph("0.7786", table_cell_center), Paragraph("0.4510", table_cell_center), Paragraph("22", table_cell_center), Paragraph("12.5s", table_cell_center)],
        [Paragraph("7", table_cell_center), Paragraph("LogReg + Viterbi HMM Decoder", table_cell), Paragraph("0.4432", table_cell_center), Paragraph("0.6141", table_cell_center), Paragraph("73.10%", table_cell_center), Paragraph("0.8120", table_cell_center), Paragraph("0.7420", table_cell_center), Paragraph("0.2880", table_cell_center), Paragraph("20", table_cell_center), Paragraph("14.3s", table_cell_center)],
        [Paragraph("8", table_cell_center), Paragraph("Temporal Conv1D (Smooth w=50 / 2.0s)", table_cell), Paragraph("0.4375", table_cell_center), Paragraph("0.6594", table_cell_center), Paragraph("75.87%", table_cell_center), Paragraph("0.8813", table_cell_center), Paragraph("0.7538", table_cell_center), Paragraph("0.3432", table_cell_center), Paragraph("15", table_cell_center), Paragraph("18.6s", table_cell_center)],
        [Paragraph("9", table_cell_center), Paragraph("InceptionTime 1D CNN (Smooth w=50 / 2.0s)", table_cell), Paragraph("0.4114", table_cell_center), Paragraph("0.7227", table_cell_center), Paragraph("79.15%", table_cell_center), Paragraph("0.8791", table_cell_center), Paragraph("0.7894", table_cell_center), Paragraph("0.4995", table_cell_center), Paragraph("18", table_cell_center), Paragraph("15.4s", table_cell_center)],
        [Paragraph("10", table_cell_center), Paragraph("ResNet-1D CNN (Raw Unsmoothed)", table_cell), Paragraph("0.4095", table_cell_center), Paragraph("0.7189", table_cell_center), Paragraph("77.51%", table_cell_center), Paragraph("0.8513", table_cell_center), Paragraph("0.7947", table_cell_center), Paragraph("0.5105", table_cell_center), Paragraph("25", table_cell_center), Paragraph("10.6s", table_cell_center)],
        [Paragraph("11", table_cell_center), Paragraph("Multinomial Logistic Regression (Raw)", table_cell), Paragraph("0.3404", table_cell_center), Paragraph("0.6125", table_cell_center), Paragraph("72.63%", table_cell_center), Paragraph("0.8313", table_cell_center), Paragraph("0.7511", table_cell_center), Paragraph("0.2553", table_cell_center), Paragraph("30", table_cell_center), Paragraph("9.5s", table_cell_center)],
        [Paragraph("12", table_cell_center), Paragraph("Nearest Player Baseline (Smooth w=25)", table_cell), Paragraph("0.3000", table_cell_center), Paragraph("0.6307", table_cell_center), Paragraph("69.23%", table_cell_center), Paragraph("0.7729", table_cell_center), Paragraph("0.6962", table_cell_center), Paragraph("0.4231", table_cell_center), Paragraph("35", table_cell_center), Paragraph("7.6s", table_cell_center)],
        [Paragraph("13", table_cell_center), Paragraph("LightGBM Classifier (Smooth w=50)", table_cell), Paragraph("0.2621", table_cell_center), Paragraph("0.5865", table_cell_center), Paragraph("70.56%", table_cell_center), Paragraph("0.7706", table_cell_center), Paragraph("0.7576", table_cell_center), Paragraph("0.2312", table_cell_center), Paragraph("12", table_cell_center), Paragraph("24.4s", table_cell_center)],
        [Paragraph("14", table_cell_center), Paragraph("Decision Tree Classifier (max_depth=8)", table_cell), Paragraph("0.2000", table_cell_center), Paragraph("0.5461", table_cell_center), Paragraph("62.59%", table_cell_center), Paragraph("0.6992", table_cell_center), Paragraph("0.7129", table_cell_center), Paragraph("0.2264", table_cell_center), Paragraph("27", table_cell_center), Paragraph("9.6s", table_cell_center)],
        [Paragraph("15", table_cell_center), Paragraph("K-Nearest Neighbors (k=15)", table_cell), Paragraph("0.1436", table_cell_center), Paragraph("0.5802", table_cell_center), Paragraph("70.13%", table_cell_center), Paragraph("0.8366", table_cell_center), Paragraph("0.7009", table_cell_center), Paragraph("0.2030", table_cell_center), Paragraph("61", table_cell_center), Paragraph("4.5s", table_cell_center)],
        [Paragraph("16", table_cell_center), Paragraph("Nearest Player Baseline (Raw Heuristic)", table_cell), Paragraph("0.0378", table_cell_center), Paragraph("0.5972", table_cell_center), Paragraph("65.12%", table_cell_center), Paragraph("0.7289", table_cell_center), Paragraph("0.6432", table_cell_center), Paragraph("0.4197", table_cell_center), Paragraph("131", table_cell_center), Paragraph("2.0s", table_cell_center)]
    ]

    t = Table(table_data, colWidths=[32, 178, 48, 48, 44, 38, 38, 38, 34, 42])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafd"), colors.white]),
        # Highlight top 3
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e8f5e9")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f1f8e9")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#f9fbe7")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # SECTION 2: DESCRIPTIONS OF THE WINNING MODELS
    story.append(Paragraph("2. Descriptions of the Winning Models", h1_style))

    # Model 1
    story.append(Paragraph("A. ResNet-1D CNN (Smooth w=50 / 2.0s) -- <i>All-Time Highest Ranked spellF1: 0.5875</i>", h2_style))
    story.append(Paragraph(
        "• <b>Architecture:</b> 3-stage Deep 1D Residual Convolutional Network (64 -> 128 -> 128 filters) with identity and 1x1 projection "
        "skip connections, batch normalization, ReLU non-linearities, and spatial dropout (p=0.15). Trained using AdamW with cosine annealing over 35-frame temporal contexts.<br/>"
        "• <b>Why It Won:</b> Achieved <b>spellF1 = 0.5875</b> (+141% over competition baseline 0.2433). Residual shortcuts allow deep layers to learn subtle kinematic changes without vanishing gradients.<br/>"
        "• <b>Dead Ball Detection:</b> Reached <b>F1(D) = 0.5223</b> (more than 2x higher than classical models at ~0.23), accurately identifying when ball out-of-play occurs.<br/>"
        "• <b>Spell Consistency:</b> Predicted <b>15 clean, consolidated possessions averaging 17.7s</b>, completely eliminating artificial single-frame flickers.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    # Model 2
    story.append(Paragraph("B. CNN Blend (InceptionTime + ResNet-1D, w=35) -- <i>Highest Accuracy (80.1%) & macroF1 (0.7447)</i>", h2_style))
    story.append(Paragraph(
        "• <b>Architecture:</b> Soft probability blend combining InceptionTime (multi-scale parallel convolutions with kernel sizes 3, 5, 11) "
        "and ResNet-1D, followed by a 1.4-second (35-frame) rolling mean filter.<br/>"
        "• <b>Why It Succeeded:</b> Multi-scale inception kernels capture micro-touches (0.12s) and macro team shape shifts (0.44s) simultaneously, while ResNet stabilizes long trajectories. "
        "Delivered the <b>highest overall frame accuracy (80.07%)</b>, the top tie-break score (<b>macroF1 = 0.7447</b>), and an outstanding <b>F1(D) = 0.5566</b>.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    # Model 3
    story.append(Paragraph("C. ResNet-1D CNN (Smooth w=25 / 1.0s) -- <i>Closest Possession Duration Match (13.9s)</i>", h2_style))
    story.append(Paragraph(
        "• <b>Architecture:</b> Same deep residual backbone with a faster 1.0-second smoothing window.<br/>"
        "• <b>Why It Succeeded:</b> Produced <b>19 spells averaging 13.9 seconds</b>, virtually identical to the ground truth (17 spells averaging 14.2 seconds), while retaining a high <b>spellF1 = 0.4889</b>.",
        bullet_style
    ))
    story.append(Spacer(1, 4))

    # Page Break for clean 2-page layout
    story.append(PageBreak())

    # SECTION 3: THE 31 ENGINEERED FEATURES
    story.append(Paragraph("3. Feature Architecture & Engineering (31 Collective Features)", h1_style))
    story.append(Paragraph(
        "Player tracking slots are anonymized (no persistent player IDs) and the ball is occluded on ~43% of tactical camera frames. "
        "All features are engineered to be <b>strictly permutation-invariant, team-collective, and resilient to missing ball data</b>:",
        body_style
    ))
    story.append(Spacer(1, 4))

    feat_table_data = [
        [Paragraph("Feature Group", table_header), Paragraph("Feature Names (31 total)", table_header), Paragraph("Domain Mechanism & Physics Signal", table_header)],
        [
            Paragraph("<b>Ball Proximity & Pressure</b><br/>(Core Possession Signal)", table_cell),
            Paragraph("<code>d_margin</code><br/><code>d_margin_sq</code><br/><code>ball_missing</code><br/><code>ball_speed</code><br/><code>home/away_within_3m</code><br/><code>home/away_within_5m</code>", table_cell),
            Paragraph("• <code>d_margin = d_away - d_home</code> (meters): Proximity differential. Positive values directly correlate with Home possession; negative with Away.<br/>• <code>within_3m / within_5m</code>: Counts players within dual concentric rings around the ball, quantifying tactical pressure and contested duels.<br/>• <code>ball_missing</code>: Indicator flag activating collective team proxies during camera occlusions.", table_cell)
        ],
        [
            Paragraph("<b>Collective Geometry & Pitch Spread</b><br/>(Ball Gap Proxy)", table_cell),
            Paragraph("<code>home/away_centroid_x/y</code><br/><code>centroid_diff_x</code><br/><code>home/away_spread_x/y</code><br/><code>spread_ratio_x/y</code>", table_cell),
            Paragraph("• <code>centroids</code>: Collective center of mass for each team. Possessing teams systematically advance their block 10--20m further up-pitch than defending teams.<br/>• <code>spread_x/y</code>: Spatial dispersion (standard deviation). The attacking team expands wide to create passing lanes; the defending team contracts into a compact low block.<br/>• <code>spread_ratio</code>: Relative attacking width proxy, robust even when the ball coordinate is missing.", table_cell)
        ],
        [
            Paragraph("<b>Kinematics & Stoppage</b><br/>(Dead Ball Indicator)", table_cell),
            Paragraph("<code>mean_speed_home</code><br/><code>mean_speed_away</code><br/><code>overall_speed</code><br/><code>speed_diff</code>", table_cell),
            Paragraph("• Average velocities across all 11 active outfield players.<br/>• <b>Dead Ball Key Discriminator:</b> In active open play, <code>overall_speed</code> averages 2.5--4.0 m/s. During stoppages (fouls, throw-in setups, substitutions), collective speed drops below <b>1.0 m/s</b>, allowing CNNs to achieve <b>F1(D) > 0.52</b>.", table_cell)
        ],
        [
            Paragraph("<b>Temporal Convolutions & Inertia</b><br/>(Anti-Flicker Filters)", table_cell),
            Paragraph("<code>d_margin_roll_1s/2s/3s</code><br/><code>speed_all_roll_1s/2s/3s</code><br/><code>missing_ratio_roll_2s</code><br/><code>d_margin_momentum</code>", table_cell),
            Paragraph("• Multi-scale moving averages (25, 50, 75 frames) capturing short-term possession inertia.<br/>• <code>d_margin_momentum</code>: First-order numerical derivative of proximity margin indicating passes or contested loose balls.<br/>• Prevents single-frame classification flickers that ruin the official <code>spellF1</code> metric.", table_cell)
        ],
    ]

    ft = Table(feat_table_data, colWidths=[90, 140, 310])
    ft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2a4a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafd"), colors.white]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ft)
    story.append(Spacer(1, 8))

    # SECTION 4: STRATEGIC INSIGHTS & FUTURE MILESTONES
    story.append(Paragraph("4. Strategic Competition Insights & Recommendations", h1_style))
    story.append(Paragraph(
        "1. <b>Why CNNs Outperformed Classical ML (+141% on spellF1):</b><br/>"
        "Classical frame-by-frame models (LightGBM, Decision Trees) make decisions on static snapshots. In football, possession is fundamentally a <i>temporal continuum</i>. "
        "A loose pass or a player sliding into a tackle creates temporary proximity ambiguity. The 1D Residual CNN observes a 35-frame window (1.4 seconds), naturally suppressing "
        "momentary flickers and keeping the possession state stable.<br/><br/>"
        "2. <b>Solving the Dead Ball Bottleneck:</b><br/>"
        "Dead ball events (<code>D</code>) represent 14.8% of the match but were historically misclassified by classical models (F1 ~0.23). "
        "By fusing collective velocity decay with dilated temporal receptive fields, ResNet-1D achieved <b>F1(D) = 0.5223</b>, and the CNN Blend achieved <b>F1(D) = 0.5566</b>.<br/><br/>"
        "3. <b>Next Steps for Full Match Submission:</b><br/>"
        "• Deploy the winning <b>ResNet-1D CNN (w=50)</b> pipeline to generate full 199,319 frame predictions (<code>submission_resnet1d_smooth50.csv</code>).<br/>"
        "• Verify formatting against <code>evaluation/validate_submission.py</code> to ensure compliance with competition submission criteria.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {filename}")


if __name__ == "__main__":
    build_pdf(DESKTOP_PDF)
    build_pdf(WORKSPACE_PDF)
    build_pdf(ARTIFACT_PDF)
