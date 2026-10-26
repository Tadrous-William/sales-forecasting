"""'How it works' image for the README and portfolio."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FB, FR = "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"
f = lambda s, b=False: ImageFont.truetype(FB if b else FR, s)  # noqa: E731
NAVY, GREEN, GREY, LINE, BG = (15, 23, 42), (4, 120, 87), (100, 116, 139), (226, 232, 240), (247, 248, 250)

if __name__ == "__main__":
    m = json.loads((ROOT / "output" / "metrics.json").read_text())
    W, H = 1600, 900
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((70, 50), "HOW IT WORKS", font=f(28, True), fill=GREEN)
    d.text((70, 92), "From sales history to a tested forecast and a stock plan", font=f(50, True), fill=NAVY)
    steps = [("1", "Data", "105 weeks of real sales (1M order lines, cleaned in SQL): weekly units for the 200 best sellers, plus the "
                           "store calendar (trading days, Christmas closure)."),
             ("2", "Signals", "Recent sales per trading day, the same weeks last year, category and store seasonality, "
                              "price changes, weeks to Christmas."),
             ("3", "Forecast", "LightGBM machine learning (expected value and low / high range), combined 50/50 with the "
                               "8-week daily average."),
             ("4", "Honest test", f"{len(m['forecast_dates'])} forecast dates in 2011, 8 weeks ahead, only past data each "
                                  "time, compared with 5 simple methods. Range checked on unseen weeks."),
             ("5", "Stock plan", "Excel + app: demand for the next 4 and 8 weeks and the stock that covers demand "
                                 "9 times out of 10, per product.")]
    bw, gap, y0 = 272, 24, 220
    for i, (n, title, text) in enumerate(steps):
        x = 70 + i * (bw + gap)
        d.rounded_rectangle([x, y0, x + bw, y0 + 470], radius=22, fill="white", outline=LINE, width=2)
        d.ellipse([x + 24, y0 + 24, x + 76, y0 + 76], fill=GREEN)
        d.text((x + 50, y0 + 50), n, font=f(28, True), fill="white", anchor="mm")
        d.text((x + 24, y0 + 100), title, font=f(33, True), fill=NAVY)
        line, yy = "", y0 + 165
        for w in text.split() + [None]:
            test = (line + " " + w).strip() if w else ""
            if w is None or d.textlength(test, font=f(22)) > bw - 44:
                d.text((x + 24, yy), line, font=f(22), fill=GREY)
                yy += 31
                line = w or ""
            else:
                line = test
    w4 = m["wape"]["store_4_weeks"]
    d.text((70, 730), f"Result: whole-store 4-week demand error {w4['combined']:.1%} vs {w4['avg_8w']:.1%} for the best simple "
                      f"method ({m['error_reduction_vs_best_simple']['store_4_weeks']:.0%} less error).", font=f(24), fill=NAVY)
    d.text((70, 770), f"{m['range_check_on_last_3_dates']['inside_low_high']:.0%} of unseen weeks fell inside the forecast range "
                      "(target: 80%).", font=f(24), fill=NAVY)
    im.save(ROOT / "portfolio" / "how_it_works.png")
    print("saved")
