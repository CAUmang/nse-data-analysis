from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data" / "bhavcopy"
DEFAULT_DAYS = 15


def build_business_days(end_date: Optional[date] = None, count: int = DEFAULT_DAYS) -> List[date]:
    """Return the last N trading days before the supplied end date."""
    if end_date is None:
        end_date = date.today()

    current_day = end_date - timedelta(days=1)
    days: List[date] = []
    while len(days) < count:
        if current_day.weekday() < 5:
            days.append(current_day)
        current_day -= timedelta(days=1)
    return days


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column_name.strip() if isinstance(column_name, str) else column_name for column_name in normalized.columns]
    return normalized


def _resolve_symbol_column(frame: pd.DataFrame) -> Optional[str]:
    for column_name in ("SYMBOL", "Symbol", "symbol"):
        if column_name in frame.columns:
            return column_name
    return None


def _resolve_series_column(frame: pd.DataFrame) -> Optional[str]:
    for column_name in ("SERIES", "Series", "series"):
        if column_name in frame.columns:
            return column_name
    return None


def _resolve_price_column(frame: pd.DataFrame, preferred: str = "CLOSE_PRICE") -> Optional[str]:
    candidate_names = [preferred, "CLOSE", "Close", "close", "Close Price", "close_price"]
    for column_name in candidate_names:
        if column_name in frame.columns:
            return column_name
    return None


def build_close_price_matrix(daily_frames: List[Tuple[date, pd.DataFrame]], price_column: str = "CLOSE_PRICE") -> pd.DataFrame:
    """Create a wide matrix of closing prices with one row per symbol and one column per day."""
    if not daily_frames:
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    for day, frame in daily_frames:
        frame = _normalize_columns(frame.copy())
        series_col = _resolve_series_column(frame)
        if series_col is not None:
            frame = frame[frame[series_col].astype(str).str.strip() == "EQ"].copy()

        symbol_col = _resolve_symbol_column(frame)
        price_col = _resolve_price_column(frame, preferred=price_column)
        if symbol_col is None or price_col is None:
            continue

        selected = frame[[symbol_col, price_col]].copy()
        selected.rename(columns={symbol_col: "SYMBOL", price_col: "CLOSE_PRICE"}, inplace=True)
        selected["date"] = day.strftime("%Y-%m-%d")
        rows.append(selected)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    combined["SYMBOL"] = combined["SYMBOL"].astype(str).str.strip()
    pivoted = combined.pivot(index="SYMBOL", columns="date", values="CLOSE_PRICE")
    return pivoted.sort_index(axis=1)


def select_stocks_by_pattern(close_price_matrix: pd.DataFrame, max_results: int = 50) -> pd.DataFrame:
    """Select stocks that show a stable first 10-day period and rising last 5-day period."""
    selected_rows: List[Dict[str, object]] = []
    date_columns = [str(column_name) for column_name in close_price_matrix.columns]
    if len(date_columns) < 15:
        return pd.DataFrame(columns=["symbol", "latest_close", "first_10_max_change_pct", "last_5_day_return_pct", "last_15_days_prices"])

    history_dates = date_columns[-15:]

    for symbol, prices in close_price_matrix.iterrows():
        series = pd.to_numeric(prices, errors="coerce").dropna()
        if len(series) < 15:
            continue

        history = series.iloc[-15:].tolist()
        stable = True
        for index in range(1, 10):
            previous_price = history[index - 1]
            current_price = history[index]
            if previous_price == 0:
                stable = False
                break
            if abs((current_price / previous_price) - 1.0) > 0.1:
                stable = False
                break

        rising = True
        for index in range(10, 15):
            previous_price = history[index - 1]
            current_price = history[index]
            if current_price <= previous_price:
                rising = False
                break

        if stable and rising:
            latest_close = history[-1]
            first_10_change = max(abs((history[index] / history[index - 1]) - 1.0) for index in range(1, 10))
            last_5_return = ((history[-1] / history[-5]) - 1.0) * 100.0 if history[-5] else float("nan")
            record: Dict[str, object] = {
                "symbol": symbol,
                "latest_close": latest_close,
                "first_10_max_change_pct": first_10_change * 100.0,
                "last_5_day_return_pct": last_5_return,
                "last_15_days_prices": history,
            }
            for date_label, price_value in zip(history_dates, history):
                record[date_label] = price_value
            selected_rows.append(record)

    if not selected_rows:
        return pd.DataFrame(columns=["symbol", "latest_close", "first_10_max_change_pct", "last_5_day_return_pct", "last_15_days_prices"])

    result = pd.DataFrame(selected_rows)
    result = result.sort_values(by=["latest_close", "last_5_day_return_pct"], ascending=[False, False])
    return result.head(max_results).reset_index(drop=True)


def _candidate_urls(day: date) -> List[str]:
    day_str = day.strftime("%d")
    month_num = day.strftime("%m")
    month_name = day.strftime("%b").upper()
    year = day.strftime("%Y")
    date_key = day.strftime("%d%m%Y")
    print("===========")
    print(date_key)
    return [
        f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_key}.csv",
        # f"https://archives.nseindia.com/content/historical/EQUITIES/{year}/{month_name}/{day_str}/cm{day_str}{month_num}{year}bhav.csv.zip",
        # f"https://www1.nseindia.com/content/historical/EQUITIES/{year}/{month_name}/{day_str}/cm{day_str}{month_num}{year}bhav.csv.zip",
    ]


def _download_with_fallback(url: str) -> Optional[requests.Response]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
    }
    try:
        response = requests.get(url, headers=headers, timeout=45, allow_redirects=True)
        if response.status_code == 200:
            return response
    except requests.exceptions.SSLError:
        try:
            response = requests.get(url, headers=headers, timeout=45, allow_redirects=True, verify=False)
            if response.status_code == 200:
                return response
        except Exception:
            return None
    except Exception:
        return None
    return None


def download_bhav_copy(day: date, output_dir: Path) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_key = day.strftime("%Y%m%d")
    target_path = output_dir / f"{date_key}.csv"
    if target_path.exists():
        return target_path

    for url in _candidate_urls(day):
        response = _download_with_fallback(url)
        if response is None:
            continue

        try:
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    csv_name = next((name for name in archive.namelist() if name.endswith(".csv")), None)
                    if csv_name is None:
                        continue
                    with archive.open(csv_name) as raw_file:
                        content = raw_file.read().decode("utf-8", errors="ignore")
                target_path.write_text(content, encoding="utf-8")
            else:
                response.content.decode("utf-8", errors="ignore")
                target_path.write_bytes(response.content)
            return target_path
        except Exception:
            continue

    return None


def load_bhav_data(days: int = DEFAULT_DAYS, output_dir: Path = DATA_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []

    for day in build_business_days(count=days):
        path = download_bhav_copy(day, output_dir)
        if path is None:
            continue
        try:
            frame = _normalize_columns(pd.read_csv(path))
        except Exception:
            continue
        frame["bhav_date"] = day.strftime("%Y-%m-%d")
        frames.append(frame)

    if not frames:
        raise RuntimeError("No bhav copy data could be downloaded. The NSE server may be unavailable or blocked.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "combined_bhav.csv", index=False)
    return combined


def summarize_bhav_data(df: pd.DataFrame) -> Dict[str, object]:
    if isinstance(df, dict):
        df = pd.DataFrame(df)

    df = _normalize_columns(df)
    total_rows = int(len(df))
    symbol_column = "SYMBOL" if "SYMBOL" in df.columns else None
    if symbol_column is None:
        unique_symbols = 0
    else:
        unique_symbols = len({str(symbol).strip() for symbol in df[symbol_column].dropna().tolist() if str(symbol).strip()})

    volume_column = "TOTTRDQTY" if "TOTTRDQTY" in df.columns else None
    if volume_column is not None:
        volume_series = pd.to_numeric(df[volume_column], errors="coerce").fillna(0)
        total_volume = int(volume_series.sum())
        if symbol_column is not None:
            volume_by_symbol = df.groupby(symbol_column)[volume_column].sum().fillna(0)
            top_volume_symbol = max(volume_by_symbol.items(), key=lambda item: (item[1], str(item[0])))[0]
        else:
            top_volume_symbol = None
    else:
        total_volume = 0
        top_volume_symbol = None

    return {
        "total_rows": total_rows,
        "unique_symbols": unique_symbols,
        "top_volume_symbol": top_volume_symbol,
        "total_volume": total_volume,
    }


def build_close_price_matrix_from_files(output_dir: Path, price_column: str = "CLOSE_PRICE") -> pd.DataFrame:
    daily_frames: List[Tuple[date, pd.DataFrame]] = []
    for csv_path in sorted(output_dir.glob("*.csv")):
        if csv_path.name == "combined_bhav.csv":
            continue
        if not csv_path.stem.isdigit() or len(csv_path.stem) != 8:
            continue
        try:
            day = date(int(csv_path.stem[:4]), int(csv_path.stem[4:6]), int(csv_path.stem[6:8]))
        except ValueError:
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        daily_frames.append((day, frame))

    return build_close_price_matrix(daily_frames, price_column=price_column)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download recent NSE bhav copies and summarize them")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of recent business days to fetch")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR), help="Directory to save bhav copies")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data = load_bhav_data(days=args.days, output_dir=output_dir)
    summary = summarize_bhav_data(data)

    close_price_matrix = build_close_price_matrix_from_files(output_dir)
    if not close_price_matrix.empty:
        close_price_matrix.to_csv(output_dir / "close_price_matrix.csv")
        selected_stocks = select_stocks_by_pattern(close_price_matrix, max_results=50)
        if not selected_stocks.empty:
            selected_stocks.to_csv(output_dir / "selected_stocks.csv", index=False)
        else:
            (output_dir / "selected_stocks.csv").write_text("", encoding="utf-8")

    print(f"Downloaded {len(data)} rows across {args.days} business days")
    print(json.dumps(summary, indent=2))
    if not close_price_matrix.empty:
        print(f"Saved closing-price matrix to {output_dir / 'close_price_matrix.csv'}")
        print(f"Saved selected stocks to {output_dir / 'selected_stocks.csv'}")


if __name__ == "__main__":
    main()
