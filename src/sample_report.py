"""3-page sample client report (PDF, < 2 MB) for the Upwork Project Catalog."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "Sample_Report_Sales_Forecast.pdf"
M = json.loads((ROOT / "output" / "metrics.json").read_text())
plan = pd.read_csv(ROOT / "output" / "stock_plan.csv")
W, H = A4
NAVY, GREEN, GREY, LIGHT = colors.HexColor("#0F172A"), colors.HexColor("#047857"), colors.HexColor("#475569"), colors.HexColor("#ECFDF5")


def header(c, page, title):
    c.setFillColor(NAVY)
    c.rect(0, H - 2.2 * cm, W, 2.2 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(1.8 * cm, H - 1.35 * cm, title)
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 1.8 * cm, H - 1.35 * cm, f"Sales forecast report · page {page} of 3")
    c.setFillColor(GREY)
    c.setFont("Helvetica", 8)
    c.drawString(1.8 * cm, 1.2 * cm, "Prepared by Tadrous William, data scientist · Sample report on public data (UCI Online Retail II)")


def para(c, x, y, text, size=10.5, width=17.4 * cm, lead=15, font="Helvetica", color=NAVY):
    c.setFont(font, size)
    c.setFillColor(color)
    for raw in text.split("\n"):
        line = ""
        for w in raw.split():
            if c.stringWidth(line + " " + w, font, size) > width:
                c.drawString(x, y, line.strip())
                y -= lead
                line = w
            else:
                line += " " + w
        c.drawString(x, y, line.strip())
        y -= lead
    return y


def kpi(c, x, y, big, label):
    c.setFillColor(LIGHT)
    c.roundRect(x, y, 5.5 * cm, 2.8 * cm, 8, fill=1, stroke=0)
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(x + 0.5 * cm, y + 1.55 * cm, big)
    para(c, x + 0.5 * cm, y + 0.95 * cm, label, size=8.5, width=4.6 * cm, lead=11, color=GREY)


w4 = M["wape"]["store_4_weeks"]
p4 = M["wape"]["product_4_weeks"]
c = canvas.Canvas(str(OUT), pagesize=A4)

# ------------------------------------------------ page 1: summary
header(c, 1, "Demand forecast & stock plan: summary")
y = para(c, 1.8 * cm, H - 3.4 * cm, "The question", size=13, font="Helvetica-Bold")
y = para(c, 1.8 * cm, y - 2, "How many units of each of our 200 best-selling products will we sell in the next 8 weeks, "
                            "and how much stock do we need so we don't run out, including over Christmas?")
y = para(c, 1.8 * cm, y - 10, "Key results", size=13, font="Helvetica-Bold")
kpi(c, 1.8 * cm, y - 3.1 * cm, f"{M['error_reduction_vs_best_simple']['store_4_weeks']:.0%} less",
    f"forecast error than the best simple method (whole store, 4 weeks ahead: {w4['combined']:.1%} vs {w4['avg_8w']:.1%})")
kpi(c, 7.75 * cm, y - 3.1 * cm, f"{M['range_check_on_last_3_dates']['inside_low_high']:.0%}",
    "of real weeks fell inside the forecast's low-high range (target 80%), tested on unseen weeks")
kpi(c, 13.7 * cm, y - 3.1 * cm, f"{plan['stock_90'].sum():,.0f}",
    "units of stock recommended for the next 4 weeks, covering demand 9 times out of 10")
y -= 4.0 * cm
y = para(c, 1.8 * cm, y, "What we found", size=13, font="Helvetica-Bold")
for b in [f"The combined forecast (machine learning + recent average) beat all 5 simple methods tested, for single products "
          f"({p4['combined']:.1%} error vs {p4['avg_8w']:.1%}) and for the whole store.",
          "Single-product weeks stay hard to predict: a few wholesale customers order thousands of units at once. "
          "That is why the stock plan uses 4-week totals and a safety range, not one number per week.",
          "Short weeks (Easter, bank holidays) and the Christmas closure are built into the forecast, so they are not "
          "mistaken for a fall in demand.",
          "Recommendation: order to the 'Stock to cover 4 weeks' column, and review products whose recent sales fall "
          "outside the forecast range."]:
    c.setFillColor(GREEN)
    c.circle(2.0 * cm, y + 3.5, 2.2, fill=1, stroke=0)
    y = para(c, 2.4 * cm, y, b, size=10, width=17 * cm, lead=14) - 5
y = para(c, 1.8 * cm, y - 6, "Data and method", size=13, font="Helvetica-Bold")
para(c, 1.8 * cm, y - 2, "105 weeks of real sales (1.07M order lines, Dec 2009 to Dec 2011), cleaned in SQL. Weekly units for "
                         "the 200 best sellers (39% of revenue). LightGBM models (expected value and low / high range) "
                         "combined 50/50 with the 8-week average per trading day. Tested on 7 forecast dates in 2011, "
                         "each using only earlier data.", size=10, lead=14)
y -= 2.6 * cm
c.drawImage(str(ROOT / "portfolio" / "how_it_works.png"), 1.6 * cm, 2.0 * cm, width=17.8 * cm, height=y - 2.0 * cm,
            preserveAspectRatio=True, anchor="n")
c.showPage()

# ------------------------------------------------ page 2: accuracy
header(c, 2, "How accurate is the forecast?")
y = para(c, 1.8 * cm, H - 3.4 * cm, f"On 7 dates in 2011 the forecast was made using only the weeks before that date, then "
                                   f"compared with what really sold over the next 8 weeks ({M['product_weeks_tested']:,} "
                                   "product-weeks). Error = total difference between forecast and actual units / actual units "
                                   "(lower is better).", size=10, lead=14)
img = ROOT / "portfolio" / "accuracy.png"
c.drawImage(str(img), 1.6 * cm, y - 8.2 * cm, width=17.8 * cm, height=6.8 * cm, preserveAspectRatio=True, anchor="c")
y -= 9.0 * cm
rows = [("Method", "Each product, 4 wk", "Whole store, 4 wk", "Bias")]
names = {"combined": "Combined forecast (used)", "avg_8w": "8-week average", "same_week_ly": "Same week last year",
         "ly_x_trend": "Last year x trend", "last_week": "Last week"}
for k, n in names.items():
    rows.append((n, f"{M['wape']['product_4_weeks'][k]:.1%}", f"{M['wape']['store_4_weeks'][k]:.1%}", f"{M['bias'][k]:+.1%}"))
xs = [1.8 * cm, 8.5 * cm, 12.3 * cm, 16.1 * cm]
for i, r in enumerate(rows):
    if i == 0 or i == 1:
        c.setFillColor(NAVY if i == 0 else LIGHT)
        c.rect(1.6 * cm, y - 4, 17.8 * cm, 16, fill=1, stroke=0)
    c.setFillColor(colors.white if i == 0 else NAVY)
    c.setFont("Helvetica-Bold" if i <= 1 else "Helvetica", 9.5)
    for x, v in zip(xs, r):
        c.drawString(x, y, v)
    y -= 18
img2 = ROOT / "portfolio" / "backtest_store.png"
c.drawImage(str(img2), 1.6 * cm, y - 7.6 * cm, width=17.8 * cm, height=7.2 * cm, preserveAspectRatio=True, anchor="n")
c.showPage()

# ------------------------------------------------ page 3: forecast + stock plan
header(c, 3, "Forecast and stock plan")
y = para(c, 1.8 * cm, H - 3.4 * cm, "Example product: weekly units sold in 2011, and the forecast for the next 8 weeks with its "
                                   "likely range (8 weeks out of 10 fall inside). The dip is the Christmas closure.", size=10, lead=14)
c.drawImage(str(ROOT / "portfolio" / "product_85123A.png"), 1.6 * cm, y - 7.2 * cm, width=17.8 * cm, height=6.8 * cm,
            preserveAspectRatio=True, anchor="c")
y -= 8.2 * cm
y = para(c, 1.8 * cm, y, "Stock plan: top-selling products (by revenue)", size=12, font="Helvetica-Bold") - 4
c.setFillColor(NAVY)
c.rect(1.6 * cm, y - 4, 17.8 * cm, 16, fill=1, stroke=0)
c.setFillColor(colors.white)
c.setFont("Helvetica-Bold", 8.5)
c.drawString(1.8 * cm, y, "Product")
for name, x in [("Sold per week", 12.4), ("Demand, next 4 wk", 15.4), ("Stock for 4 wk (90%)", 19.2)]:
    c.drawRightString(x * cm, y, name)
y -= 17
c.setFont("Helvetica", 8.5)
for i, r in enumerate(plan[plan["avg_week_last_8"] > 0].head(10).itertuples()):
    if i % 2:
        c.setFillColor(colors.HexColor("#F8FAFC"))
        c.rect(1.6 * cm, y - 4, 17.8 * cm, 15, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.drawString(1.8 * cm, y, r.product_name.title()[:44])
    c.drawRightString(12.4 * cm, y, f"{r.avg_week_last_8:,.0f}")
    c.drawRightString(15.4 * cm, y, f"{r.demand_4w:,.0f}")
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawRightString(19.2 * cm, y, f"{r.stock_90:,.0f}")
    c.setFont("Helvetica", 8.5)
    y -= 15
y = para(c, 1.8 * cm, y - 8, "Sold per week = average of the last 8 weeks. The Excel file the client gets (below) has all "
                             "200 products, the weekly forecast with low / high numbers, and the accuracy test.",
         size=9.5, lead=13, color=GREY)
c.drawImage(str(ROOT / "portfolio" / "excel.png"), 1.6 * cm, 2.0 * cm, width=17.8 * cm, height=y - 2.4 * cm,
            preserveAspectRatio=True, anchor="n")
c.save()
print(OUT, round(OUT.stat().st_size / 1024), "KB")
