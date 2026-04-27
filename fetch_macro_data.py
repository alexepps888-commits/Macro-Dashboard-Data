"""
fetch_macro_data.py

获取美国和中国基础宏观经济数据，并输出为 CSV / JSON。

依赖：
    pip install pandas pandas_datareader akshare

说明：
- 美国数据：来自 FRED，使用 pandas_datareader。
- 中国数据：来自 AKShare，优先获取 GDP 和 LPR。
- 所有抓取函数都有异常处理，单个数据源失败不会导致程序崩溃。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas_datareader import data as pdr


# =========================
# 基础配置
# =========================

OUTPUT_DIR = Path("data")
OUTPUT_CSV = OUTPUT_DIR / "macro_latest.csv"
OUTPUT_JSON = OUTPUT_DIR / "macro_latest.json"

# FRED 拉取区间不用太长，只要能覆盖最新一期即可。
# GDP 是季度数据，CPI 和联邦基金利率是月度数据。
FRED_START_DATE = "2015-01-01"

FRED_SERIES = {
    "GDP": {
        "country": "United States",
        "country_cn": "美国",
        "indicator": "GDP",
        "indicator_cn": "国内生产总值",
        "unit": "Billions of USD, SAAR",
        "unit_cn": "十亿美元，季调年率",
        "frequency": "Quarterly",
        "source": "FRED / U.S. Bureau of Economic Analysis",
    },
    "CPIAUCSL": {
        "country": "United States",
        "country_cn": "美国",
        "indicator": "CPI",
        "indicator_cn": "消费者物价指数",
        "unit": "Index 1982-1984=100, SA",
        "unit_cn": "指数，1982-1984=100，季调",
        "frequency": "Monthly",
        "source": "FRED / U.S. Bureau of Labor Statistics",
    },
    "FEDFUNDS": {
        "country": "United States",
        "country_cn": "美国",
        "indicator": "Federal Funds Effective Rate",
        "indicator_cn": "联邦基金有效利率",
        "unit": "Percent",
        "unit_cn": "%",
        "frequency": "Monthly",
        "source": "FRED / Federal Reserve Board",
    },
}


# =========================
# 日志
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# =========================
# 通用工具函数
# =========================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    """尽量把各种数值格式转成 float，失败则返回 None。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def safe_str(value: Any) -> Optional[str]:
    """把日期/字符串安全转成字符串。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return str(value)


def make_record(
    *,
    country: str,
    country_cn: str,
    indicator: str,
    indicator_cn: str,
    value: Optional[float],
    unit: str,
    unit_cn: str,
    period: Optional[str],
    frequency: str,
    source: str,
    series_id: Optional[str] = None,
    status: str = "ok",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "country": country,
        "country_cn": country_cn,
        "indicator": indicator,
        "indicator_cn": indicator_cn,
        "value": value,
        "unit": unit,
        "unit_cn": unit_cn,
        "period": period,
        "frequency": frequency,
        "source": source,
        "series_id": series_id,
        "status": status,
        "error": error,
        "fetched_at_utc": now_utc_iso(),
    }


def make_error_record(
    *,
    country: str,
    country_cn: str,
    indicator: str,
    indicator_cn: str,
    source: str,
    error: Exception | str,
    series_id: Optional[str] = None,
) -> Dict[str, Any]:
    return make_record(
        country=country,
        country_cn=country_cn,
        indicator=indicator,
        indicator_cn=indicator_cn,
        value=None,
        unit="",
        unit_cn="",
        period=None,
        frequency="",
        source=source,
        series_id=series_id,
        status="failed",
        error=str(error),
    )


# =========================
# FRED：美国宏观数据
# =========================

def fetch_latest_fred_series(series_id: str, meta: Dict[str, str]) -> Dict[str, Any]:
    """
    从 FRED 获取某个序列的最新一期非空数据。
    单个序列失败时返回 failed record，不抛出异常。
    """
    try:
        df = pdr.DataReader(series_id, "fred", start=FRED_START_DATE)

        if df.empty or series_id not in df.columns:
            raise ValueError(f"FRED returned empty data for {series_id}")

        series = df[series_id].dropna()

        if series.empty:
            raise ValueError(f"No non-null observations for {series_id}")

        latest_date = series.index[-1]
        latest_value = safe_float(series.iloc[-1])

        return make_record(
            country=meta["country"],
            country_cn=meta["country_cn"],
            indicator=meta["indicator"],
            indicator_cn=meta["indicator_cn"],
            value=latest_value,
            unit=meta["unit"],
            unit_cn=meta["unit_cn"],
            period=latest_date.date().isoformat(),
            frequency=meta["frequency"],
            source=meta["source"],
            series_id=series_id,
        )

    except Exception as exc:
        logging.warning("Failed to fetch FRED series %s: %s", series_id, exc)
        return make_error_record(
            country=meta["country"],
            country_cn=meta["country_cn"],
            indicator=meta["indicator"],
            indicator_cn=meta["indicator_cn"],
            source=meta["source"],
            series_id=series_id,
            error=exc,
        )


def fetch_us_macro_from_fred() -> List[Dict[str, Any]]:
    records = []
    for series_id, meta in FRED_SERIES.items():
        records.append(fetch_latest_fred_series(series_id, meta))
    return records


# =========================
# AKShare：中国宏观数据
# =========================

def import_akshare():
    """
    动态导入 akshare。
    这样即使用户没有安装 akshare，脚本也不会在 import 阶段崩溃。
    """
    try:
        import akshare as ak
        return ak, None
    except Exception as exc:
        return None, exc


def find_date_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """从候选名称中寻找日期列。"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def latest_row_by_date(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    按日期列排序后返回最新一行。
    如果日期解析失败，则尽量返回原始最后一行。
    """
    tmp = df.copy()
    tmp["_parsed_date"] = pd.to_datetime(tmp[date_col], errors="coerce")

    if tmp["_parsed_date"].notna().any():
        tmp = tmp.sort_values("_parsed_date")
        return tmp.iloc[-1]

    return df.iloc[-1]


def fetch_china_gdp_from_akshare() -> List[Dict[str, Any]]:
    """
    获取中国 GDP 及 GDP 同比增速。
    AKShare 接口：macro_china_gdp
    常见字段：
        - 季度
        - 国内生产总值-绝对值，单位：亿元
        - 国内生产总值-同比增长，单位：%
    """
    ak, import_error = import_akshare()

    if ak is None:
        return [
            make_error_record(
                country="China",
                country_cn="中国",
                indicator="GDP",
                indicator_cn="国内生产总值",
                source="AKShare",
                error=f"akshare import failed: {import_error}",
                series_id="macro_china_gdp",
            )
        ]

    try:
        df = ak.macro_china_gdp()

        if df is None or df.empty:
            raise ValueError("AKShare macro_china_gdp returned empty data")

        date_col = find_date_column(df, ["季度", "日期", "月份", "时间"])
        if date_col is None:
            raise ValueError(f"Cannot find date column in columns: {list(df.columns)}")

        latest = latest_row_by_date(df, date_col)
        period = safe_str(latest.get(date_col))

        gdp_value_col = "国内生产总值-绝对值"
        gdp_yoy_col = "国内生产总值-同比增长"

        records: List[Dict[str, Any]] = []

        if gdp_value_col in df.columns:
            records.append(
                make_record(
                    country="China",
                    country_cn="中国",
                    indicator="GDP",
                    indicator_cn="国内生产总值",
                    value=safe_float(latest.get(gdp_value_col)),
                    unit="100 million CNY",
                    unit_cn="亿元人民币",
                    period=period,
                    frequency="Quarterly",
                    source="AKShare / Eastmoney",
                    series_id="macro_china_gdp",
                )
            )
        else:
            raise ValueError(f"Missing expected column: {gdp_value_col}")

        if gdp_yoy_col in df.columns:
            records.append(
                make_record(
                    country="China",
                    country_cn="中国",
                    indicator="GDP YoY",
                    indicator_cn="GDP同比增速",
                    value=safe_float(latest.get(gdp_yoy_col)),
                    unit="Percent",
                    unit_cn="%",
                    period=period,
                    frequency="Quarterly",
                    source="AKShare / Eastmoney",
                    series_id="macro_china_gdp",
                )
            )

        return records

    except Exception as exc:
        logging.warning("Failed to fetch China GDP from AKShare: %s", exc)
        return [
            make_error_record(
                country="China",
                country_cn="中国",
                indicator="GDP",
                indicator_cn="国内生产总值",
                source="AKShare / Eastmoney",
                series_id="macro_china_gdp",
                error=exc,
            )
        ]


def fetch_china_lpr_from_akshare() -> List[Dict[str, Any]]:
    """
    获取中国 LPR 利率。
    AKShare 接口：macro_china_lpr
    常见字段：
        - TRADE_DATE
        - LPR1Y，单位：%
        - LPR5Y，单位：%
    """
    ak, import_error = import_akshare()

    if ak is None:
        return [
            make_error_record(
                country="China",
                country_cn="中国",
                indicator="LPR",
                indicator_cn="贷款市场报价利率",
                source="AKShare",
                error=f"akshare import failed: {import_error}",
                series_id="macro_china_lpr",
            )
        ]

    try:
        df = ak.macro_china_lpr()

        if df is None or df.empty:
            raise ValueError("AKShare macro_china_lpr returned empty data")

        date_col = find_date_column(df, ["TRADE_DATE", "日期", "时间"])
        if date_col is None:
            raise ValueError(f"Cannot find date column in columns: {list(df.columns)}")

        # 只保留至少有一个 LPR 字段非空的记录，避免历史老数据中 LPR 为空。
        lpr_cols = [col for col in ["LPR1Y", "LPR5Y"] if col in df.columns]
        if not lpr_cols:
            raise ValueError(f"Cannot find LPR columns in columns: {list(df.columns)}")

        valid_df = df.dropna(subset=lpr_cols, how="all")
        if valid_df.empty:
            raise ValueError("No valid LPR observations found")

        latest = latest_row_by_date(valid_df, date_col)
        period = safe_str(latest.get(date_col))

        records: List[Dict[str, Any]] = []

        if "LPR1Y" in valid_df.columns:
            records.append(
                make_record(
                    country="China",
                    country_cn="中国",
                    indicator="1Y LPR",
                    indicator_cn="1年期贷款市场报价利率",
                    value=safe_float(latest.get("LPR1Y")),
                    unit="Percent",
                    unit_cn="%",
                    period=period,
                    frequency="Monthly",
                    source="AKShare / Eastmoney",
                    series_id="macro_china_lpr",
                )
            )

        if "LPR5Y" in valid_df.columns:
            records.append(
                make_record(
                    country="China",
                    country_cn="中国",
                    indicator="5Y LPR",
                    indicator_cn="5年期贷款市场报价利率",
                    value=safe_float(latest.get("LPR5Y")),
                    unit="Percent",
                    unit_cn="%",
                    period=period,
                    frequency="Monthly",
                    source="AKShare / Eastmoney",
                    series_id="macro_china_lpr",
                )
            )

        return records

    except Exception as exc:
        logging.warning("Failed to fetch China LPR from AKShare: %s", exc)
        return [
            make_error_record(
                country="China",
                country_cn="中国",
                indicator="LPR",
                indicator_cn="贷款市场报价利率",
                source="AKShare / Eastmoney",
                series_id="macro_china_lpr",
                error=exc,
            )
        ]


def fetch_china_macro_from_akshare() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    records.extend(fetch_china_gdp_from_akshare())
    records.extend(fetch_china_lpr_from_akshare())
    return records


# =========================
# 输出
# =========================

def save_outputs(records: List[Dict[str, Any]]) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    # 统一列顺序，便于前端读取。
    preferred_columns = [
        "country",
        "country_cn",
        "indicator",
        "indicator_cn",
        "value",
        "unit",
        "unit_cn",
        "period",
        "frequency",
        "source",
        "series_id",
        "status",
        "error",
        "fetched_at_utc",
    ]

    for col in preferred_columns:
        if col not in df.columns:
            df[col] = None

    df = df[preferred_columns]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return df


def main() -> None:
    all_records: List[Dict[str, Any]] = []

    logging.info("Fetching U.S. macro data from FRED...")
    all_records.extend(fetch_us_macro_from_fred())

    logging.info("Fetching China macro data from AKShare...")
    all_records.extend(fetch_china_macro_from_akshare())

    df = save_outputs(all_records)

    ok_count = int((df["status"] == "ok").sum())
    failed_count = int((df["status"] == "failed").sum())

    logging.info("Done. OK: %s, Failed: %s", ok_count, failed_count)
    logging.info("CSV saved to: %s", OUTPUT_CSV)
    logging.info("JSON saved to: %s", OUTPUT_JSON)

    print("\nLatest macro data:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
