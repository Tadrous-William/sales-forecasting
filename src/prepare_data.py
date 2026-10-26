"""Step 1: weekly sales per product from the cleaned online-store data (UCI Online Retail II, SQL-cleaned in the
Power BI project), plus the store calendar (trading days per week, Christmas closure).

Output (data/): weekly_product.parquet, products.parquet, weekly_total.parquet
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("D:/Portfolio_Projects/powerbi-sales-dashboard/data/sales.duckdb")
DATA = ROOT / "data"
N_PRODUCTS = 200
LAST_FULL_WEEK = "2011-11-28"  # the data stops on Fri 9 Dec 2011, so the week of 5 Dec is incomplete

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    c = duckdb.connect(str(SOURCE), read_only=True)
    weeks = pd.DataFrame({"week": pd.date_range("2009-11-30", LAST_FULL_WEEK, freq="W-MON")})
    days = c.sql("""select date_trunc('week', order_date) as week, count(distinct order_date) as trading_days
                    from fact_sales group by 1""").df()
    weeks = weeks.merge(days, on="week", how="left").fillna({"trading_days": 0})
    weeks["trading_days"] = weeks["trading_days"].astype(int)

    products = c.sql(f"""
        with last_year as (
            select stock_code, sum(revenue) as revenue, count(distinct date_trunc('week', order_date)) as active_weeks
            from fact_sales where order_date >= date '2010-11-29' and order_date < date '2011-12-05' group by 1)
        select l.stock_code, p.product_name, p.category, round(l.revenue, 2) as revenue_last_52w, l.active_weeks
        from last_year l join dim_product p using (stock_code)
        where l.active_weeks >= 40
        order by l.revenue desc limit {N_PRODUCTS}""").df()
    c.register("chosen", products[["stock_code"]])
    sales = c.sql(f"""
        select date_trunc('week', order_date) as week, stock_code, sum(quantity) as units, sum(revenue) as revenue
        from fact_sales join chosen using (stock_code)
        where order_date < date '{LAST_FULL_WEEK}' + interval 7 day
        group by 1, 2""").df()
    first = sales.groupby("stock_code")["week"].min().rename("first_week")
    grid = weeks.merge(products[["stock_code"]], how="cross")
    grid = grid.merge(first, on="stock_code")
    grid = grid[grid["week"] >= grid["first_week"]].drop(columns="first_week")  # a product exists from its first sale
    wp = grid.merge(sales, on=["week", "stock_code"], how="left").fillna({"units": 0, "revenue": 0})
    wp["avg_price"] = (wp["revenue"] / wp["units"]).where(wp["units"] > 0)
    wp = wp.sort_values(["stock_code", "week"]).reset_index(drop=True)

    total = c.sql(f"""select date_trunc('week', order_date) as week, sum(revenue) as revenue, sum(quantity) as units
                      from fact_sales where order_date < date '{LAST_FULL_WEEK}' + interval 7 day group by 1""").df()
    total = weeks.merge(total, on="week", how="left").fillna({"revenue": 0, "units": 0})

    wp.to_parquet(DATA / "weekly_product.parquet", index=False)
    products.to_parquet(DATA / "products.parquet", index=False)
    total.to_parquet(DATA / "weekly_total.parquet", index=False)
    share = wp["revenue"].sum() / total["revenue"].sum()
    print(f"{len(weeks)} weeks ({weeks.week.min():%d %b %Y} to {weeks.week.max():%d %b %Y}), "
          f"{wp.stock_code.nunique()} products = {share:.0%} of store revenue, {len(wp):,} product-weeks; "
          f"closed weeks: {(weeks.trading_days == 0).sum()}")
