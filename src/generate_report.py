from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Preformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors

import os
import json
import datetime


# ============================================================
# House Prices – Advanced Regression Techniques
# Projektni zadatak – Tehnička dokumentacija (PDF)
# ============================================================

def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _register_unicode_font(styles) -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
        "/Library/Fonts/DejaVu Sans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("UnicodeFont", path))
                for name in ["Title", "Heading1", "Heading2", "Heading3", "Normal", "BodyText", "Italic"]:
                    if name in styles:
                        styles[name].fontName = "UnicodeFont"
                return "UnicodeFont"
            except Exception:
                pass
    return styles["Normal"].fontName


def _safe_read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _tree(root: str, max_depth: int = 3) -> str:
    root = os.path.abspath(root)
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", ".idea", "__MACOSX"}]
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue

        indent = "  " * depth
        if rel == ".":
            lines.append("house_prices_project/")
        else:
            lines.append(f"{indent}{os.path.basename(dirpath)}/")

        for fn in sorted([f for f in filenames if f not in {".DS_Store"}]):
            lines.append(f"{indent}  {fn}")
    return "\n".join(lines)


def _add_style_safe(styles, style: ParagraphStyle):
    if style.name in styles:
        return
    styles.add(style)


def _fmt_float(x):
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


class TOCDocTemplate(SimpleDocTemplate):
    """
    Custom doc template that records H1/H2 into TableOfContents.
    IMPORTANT: ReportLab's TableOfContents expects TOCEntry tuples as (level, text, pageNumber).
    """
    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style_name = getattr(flowable.style, "name", "")
        if style_name not in ("H1", "H2"):
            return

        level = 0 if style_name == "H1" else 1
        text = flowable.getPlainText()

        # MUST be 3-tuple for compatibility
        self.notify("TOCEntry", (level, text, self.page))



def build_report():
    # ---------- Metadata ----------
    faculty = "Fakultet tehničkih nauka"
    course = "Metode istraživanja i eksploatacije podataka"
    work_type = "Projektni zadatak"
    title = "House Prices – Advanced Regression Techniques"
    author = "Dunja Cvjetinović"
    index_no = "IT2/2020"
    year = "2026"
    gen_date = datetime.date.today().isoformat()
    author_line = f"{author} ({index_no})"

    # ---------- Paths ----------
    root = _project_root()
    report_path = os.path.join(root, "results", "House_Prices_Report.pdf")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # ---------- Document ----------
    doc = TOCDocTemplate(
        report_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()
    _register_unicode_font(styles)

    # ---------- Styles ----------
    _add_style_safe(styles, ParagraphStyle(
        name="CoverCenter",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=12,
        leading=16,
        spaceAfter=6
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=13,
        leading=18,
        spaceAfter=10
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        alignment=TA_JUSTIFY,
        fontSize=11,
        leading=15,
        spaceAfter=8
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="Caption",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        textColor=colors.grey,
        spaceAfter=10
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="SmallMono",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="TOCTitle",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        spaceAfter=12
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="H1",
        parent=styles["Heading1"],
        spaceBefore=0,
        spaceAfter=10
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        spaceBefore=6,
        spaceAfter=8
    ))
    _add_style_safe(styles, ParagraphStyle(
        name="H3",
        parent=styles["Heading3"],
        spaceBefore=6,
        spaceAfter=6
    ))

    # ---------- TOC ----------
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOCLevel0",
            parent=styles["BodyText"],
            fontSize=11,
            leading=14,
            leftIndent=0,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        ParagraphStyle(
            name="TOCLevel1",
            parent=styles["BodyText"],
            fontSize=10,
            leading=13,
            leftIndent=18,
            firstLineIndent=0,
            spaceAfter=4,
        ),
    ]

    story = []

    # COVER
    story.append(Spacer(1, 1.2 * inch))

    story.append(Paragraph(f"<b>{faculty}</b>", styles["CoverCenter"]))
    story.append(Paragraph(f"<b>Predmet:</b> {course}", styles["CoverCenter"]))

    story.append(Spacer(1, 1.3 * inch))

    story.append(Paragraph(work_type, styles["CoverSubtitle"]))
    story.append(Paragraph(title, styles["CoverTitle"]))

    story.append(Spacer(1, 2.8 * inch)) 

    story.append(Paragraph(f"<b>{author} ({index_no})</b>", styles["CoverCenter"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Datum generisanja dokumenta: {gen_date}", styles["CoverCenter"]))

    story.append(PageBreak())

    # TOC
    story.append(Paragraph("Sadržaj", styles["TOCTitle"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(toc)
    story.append(PageBreak())

    # 1
    story.append(Paragraph("1. Pregled projekta", styles["H1"]))
    story.append(Paragraph(
        "Ovaj dokument predstavlja tehničku dokumentaciju projekta razvijenog u okviru Kaggle takmičenja "
        "<b>House Prices: Advanced Regression Techniques</b>. Problem pripada oblasti <b>supervised machine learning</b> "
        "(regresija), gde je cilj predviđanje ciljne promenljive <b>SalePrice</b> (tržišna cena kuće) "
        "na osnovu numeričkih i kategorijskih karakteristika objekta.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Ulaz:</b> train.csv i test.csv (kombinacija numeričkih i kategorijskih atributa).<br/>"
        "<b>Izlaz:</b> submission.csv sa predikcijama SalePrice za test skup.<br/>"
        "<b>Metrika:</b> RMSE nad log-transformisanom ciljnom promenljivom (log1p), u skladu sa praksom takmičenja.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Radi stabilizacije varijanse i smanjenja uticaja ekstremnih vrednosti, ciljna promenljiva se posmatra u log "
        "prostoru. Projekat je organizovan modularno kako bi se obezbedila reprodukcija: učitavanje podataka, "
        "feature engineering, preprocesiranje, trening i evaluacija, predikcija, vizualizacije i izveštavanje.",
        styles["Body"]
    ))
    story.append(Paragraph("Korišćene biblioteke", styles["H2"]))
    story.append(Paragraph(
        "pandas, numpy, scikit-learn, xgboost, optuna, joblib, matplotlib/seaborn, reportlab.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # 2
    story.append(Paragraph("2. Arhitektura sistema", styles["H1"]))
    story.append(Paragraph(
        "Projekat je organizovan tako da razdvoji ulazne podatke, izvorni kod, modele i rezultate. "
        "Ovakva struktura olakšava održavanje i ponovljivo pokretanje celog pipeline-a.",
        styles["Body"]
    ))
    story.append(Paragraph("Struktura projekta (folder tree)", styles["H2"]))
    story.append(Preformatted(_tree(root, max_depth=3), styles["SmallMono"]))
    story.append(Paragraph(
        "<b>data/</b> sadrži ulazne skupove train/test i primer submission fajla.<br/>"
        "<b>src/</b> sadrži implementaciju pipeline-a (preprocess, trening, predikcija, vizualizacije).<br/>"
        "<b>models/</b> sadrži sačuvane modele.<br/>"
        "<b>results/</b> sadrži grafike, CV rezultate, submission fajlove i generisani PDF izveštaj.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # 3
    story.append(Paragraph("3. Tok obrade (Pipeline)", styles["H1"]))
    story.append(Paragraph(
        "Tipičan tok obrade u ML projektu obuhvata: učitavanje podataka → čišćenje/outlieri → feature engineering → "
        "preprocesiranje (missing values, encoding, scaling) → treniranje i validacija (cross-validation) → izbor "
        "najboljeg modela → treniranje na celom skupu → predikcija i izvoz submission fajla.",
        styles["Body"]
    ))
    story.append(Paragraph("Implementacija u ovom projektu", styles["H2"]))
    story.append(Paragraph(
        "<b>Učitavanje:</b> src/load_data.py<br/>"
        "<b>Outlieri + feature engineering:</b> src/feature_engineering.py<br/>"
        "<b>Preprocesiranje:</b> src/utils.py<br/>"
        "<b>Trening + CV:</b> src/train_advanced.py, src/train_search.py (Optuna)<br/>"
        "<b>Finalni trening:</b> src/train_final_models.py (na osnovu results/cv_results.json)<br/>"
        "<b>Predikcija:</b> src/predict_final.py<br/>"
        "<b>Vizualizacije:</b> src/visualize.py<br/>"
        "<b>Dokumentacija:</b> src/generate_report.py",
        styles["Body"]
    ))
    story.append(PageBreak())

    # 4
    story.append(Paragraph("4. Moduli i funkcije", styles["H1"]))
    story.append(Paragraph(
        "U nastavku je kratak opis najvažnijih modula. Fokus je na odgovornostima modula i ulaz/izlaz fajlovima, "
        "kako bi projekat bio proverljiv i reproduktivan.",
        styles["Body"]
    ))
    modules = [
        ("load_data.py", "Učitava train/test podatke iz data/ i vraća DataFrame-ove."),
        ("feature_engineering.py", "Uklanjanje outliera i kreiranje izvedenih karakteristika."),
        ("utils.py", "Preprocesiranje (missing values, encoding) i pomoćne funkcije."),
        ("train_advanced.py", "Trening Ridge/ElasticNet/XGBoost modela uz K-Fold CV i upis CV metrike."),
        ("train_search.py", "Optuna pretraga hiperparametara (XGBoost i opciono LGBM/CatBoost)."),
        ("train_final_models.py", "Bira najbolji model na osnovu cv_results.json i trenira na punom train skupu."),
        ("predict_final.py", "Učitava finalni model i generiše submission_final.csv."),
        ("visualize.py", "Generiše EDA grafike i (ako postoji) feature importance grafike."),
        ("generate_report.py", "Generiše PDF dokumentaciju i ubacuje grafike iz results/."),
    ]
    for mod, desc in modules:
        story.append(Paragraph(f"<b>{mod}</b> – {desc}", styles["Body"]))
    story.append(PageBreak())

    # 5
    story.append(Paragraph("5. Modeli i evaluacija", styles["H1"]))
    story.append(Paragraph(
        "U projektu se koriste linearni modeli (Ridge, ElasticNet) i gradient boosting model (XGBoost). "
        "Evaluacija se radi pomoću K-Fold cross-validation pristupa, a metrike se čuvaju u JSON fajlu. "
        "Niži RMSE označava bolju generalizaciju, dok <i>std</i> predstavlja varijabilnost rezultata po fold-ovima "
        "(manji std → stabilniji model).",
        styles["Body"]
    ))

    cv_path = os.path.join(root, "results", "cv_results.json")
    cv = _safe_read_json(cv_path)

    if isinstance(cv, dict) and cv:
        story.append(Paragraph("CV rezultati (iz results/cv_results.json)", styles["H2"]))
        lines = []
        best_name = None
        best_rmse = None
        for name, val in cv.items():
            if isinstance(val, list) and len(val) >= 2:
                mean_rmse, std_rmse = val[0], val[1]
                lines.append(f"- {name}: RMSE = {_fmt_float(mean_rmse)} (std = {_fmt_float(std_rmse)})")
                if best_rmse is None or float(mean_rmse) < float(best_rmse):
                    best_rmse = mean_rmse
                    best_name = name
        story.append(Preformatted("\n".join(lines), styles["SmallMono"]))
        if best_name is not None:
            story.append(Paragraph(
                f"Na osnovu CV rezultata, najbolji model po RMSE metriki je: <b>{best_name}</b>.",
                styles["Body"]
            ))
    else:
        story.append(Paragraph(
            "CV rezultati nisu pronađeni (results/cv_results.json). Pokreni trening skripte da bi se generisali.",
            styles["Body"]
        ))
    story.append(PageBreak())

    # 6
    story.append(Paragraph("6. Rezultati i artefakti", styles["H1"]))
    story.append(Paragraph("Tokom rada, projekat generiše sledeće artefakte:", styles["Body"]))
    story.append(Paragraph(
        "• <b>models/</b>: sačuvani modeli (npr. final_model.pkl, elasticnet_model.pkl, xgboost_model.json).<br/>"
        "• <b>results/</b>: submission fajlovi, grafici, cv_results.json i ovaj PDF dokument.",
        styles["Body"]
    ))

    story.append(Paragraph("Vizualizacije", styles["H2"]))
    images = [
        ("Distribucija cena (train)", os.path.join(root, "results", "plot_saleprice_distribution.png"),
         "Distribucija ciljne promenljive je tipično desno asimetrična (dug rep), što opravdava log transformaciju."),
        ("Top korelacije sa cenom", os.path.join(root, "results", "plot_top_correlations.png"),
         "Prikaz najkorelisanijih atributa pomaže u identifikaciji ključnih faktora koji utiču na cenu."),
        ("Heatmap korelacija", os.path.join(root, "results", "plot_heatmap.png"),
         "Heatmap prikazuje međusobne veze atributa i može ukazati na multikolinearnost."),
        ("Feature importance (XGBoost)", os.path.join(root, "results", "plot_feature_importance.png"),
         "Ako je dostupno, prikazuje relativnu važnost karakteristika prema XGBoost modelu."),
    ]
    for title_img, path, caption in images:
        if os.path.exists(path):
            story.append(Paragraph(title_img, styles["H3"]))
            story.append(Spacer(1, 0.06 * inch))
            story.append(Image(path, width=5.7 * inch, height=3.3 * inch))
            story.append(Spacer(1, 0.04 * inch))
            story.append(Paragraph(caption, styles["Caption"]))
    story.append(PageBreak())

    story.append(Paragraph("7. Rezultat na Kaggle-u", styles["H1"]))

    score_path = os.path.join(root, "results", "kaggle_score.png")
    if os.path.exists(score_path):
        story.append(Paragraph(
            "U nastavku je prikazan screenshot rezultata (score) sa Kaggle platforme nakon predaje submission fajla.",
            styles["Body"]
        ))
        story.append(Spacer(1, 0.08 * inch))
        story.append(Image(score_path, width=5.9 * inch, height=3.4 * inch))
        story.append(Spacer(1, 0.06 * inch))
        story.append(Paragraph(
            "Slika: Prikaz rezultata na Kaggle-u (My Submissions).",
            styles["Caption"]
        ))
    else:
        story.append(Paragraph(
            "Screenshot rezultata nije pronađen. Sačuvaj sliku kao results/kaggle_score.png i ponovo generiši PDF.",
            styles["Body"]
        ))
    story.append(PageBreak())

    # 7
    story.append(Paragraph("8. Kako koristiti (reprodukcija)", styles["H1"]))
    story.append(Paragraph(
        "Iz root foldera projekta moguće je pokrenuti kompletnu obradu sledećim komandama:",
        styles["Body"]
    ))
    cmds = """pip install -r requirements.txt

python3 -m src.train_advanced
python3 -m src.train_search --n_trials 30
python3 -m src.train_final_models
python3 -m src.predict_final
python3 -m src.visualize
python3 -m src.generate_report
"""
    story.append(Preformatted(cmds, styles["SmallMono"]))
    story.append(Paragraph(
        "<i>Napomena:</i> Alternativno, može se koristiti <b>python3 main.py --all</b> za pokretanje kompletne obrade.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # 8
    story.append(Paragraph("9. Zaključak i moguća unapređenja", styles["H1"]))
    story.append(Paragraph(
        "Projekat implementira kompletan ML pipeline za regresioni zadatak: priprema podataka, feature engineering, "
        "preprocesiranje, trening i evaluacija modela, generisanje predikcija i izvoz Kaggle submission fajla. "
        "Modularna struktura koda omogućava jednostavno proširenje i reprodukciju rezultata.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "Moguća unapređenja uključuju: naprednije enkodovanje kategorijskih promenljivih (npr. target encoding), "
        "ensemble pristupe (stacking/blending), finiju optimizaciju hiperparametara uz kontrolu overfitting-a, "
        "kao i detaljniju analizu reziduala i grešaka modela.",
        styles["Body"]
    ))
    story.append(Paragraph("<i>Generisano automatski pomoću Python-a (ReportLab).</i>", styles["Body"]))

    doc.multiBuild(story, maxPasses=2)

    print(f"📄 PDF izveštaj uspešno kreiran: {report_path}")


if __name__ == "__main__":
    build_report()