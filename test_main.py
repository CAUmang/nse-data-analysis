import unittest
from datetime import date

import pandas as pd

from main import build_business_days, build_close_price_matrix, select_stocks_by_pattern, summarize_bhav_data


class BhavScriptTests(unittest.TestCase):
    def test_build_business_days_skips_weekends(self):
        days = build_business_days(end_date=date(2024, 1, 5), count=3)
        self.assertEqual(days, [date(2024, 1, 4), date(2024, 1, 3), date(2024, 1, 2)])

    def test_summarize_bhav_data_uses_common_columns(self):
        df = {
            "SYMBOL": ["A", "A", "B"],
            "CLOSE": [100.0, 110.0, 90.0],
            "TOTTRDQTY": [1000, 2000, 3000],
            "TOTTRDVAL": [10000.0, 22000.0, 27000.0],
        }
        summary = summarize_bhav_data(df)
        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["unique_symbols"], 2)
        self.assertEqual(summary["top_volume_symbol"], "B")
        self.assertAlmostEqual(summary["total_volume"], 6000)

    def test_build_close_price_matrix_and_pattern_selection(self):
        daily_frames = []
        for day_index, prices in enumerate([
            [100.0, 102.0],
            [100.5, 101.0],
            [101.0, 100.5],
            [100.8, 99.0],
            [101.0, 98.5],
            [101.2, 97.0],
            [101.3, 96.0],
            [101.4, 95.0],
            [101.5, 94.0],
            [101.6, 93.0],
            [102.0, 92.0],
            [103.0, 93.0],
            [104.0, 94.0],
            [105.0, 95.0],
            [106.0, 96.0],
        ]):
            frame = pd.DataFrame({"SYMBOL": ["A", "B"], "CLOSE_PRICE": prices})
            daily_frames.append((date(2024, 1, 1 + day_index), frame))

        matrix = build_close_price_matrix(daily_frames)
        self.assertEqual(matrix.loc["A", "2024-01-01"], 100.0)
        self.assertEqual(matrix.loc["B", "2024-01-15"], 96.0)

        selected = select_stocks_by_pattern(matrix, max_results=10)
        self.assertEqual(selected.iloc[0]["symbol"], "A")
        self.assertEqual(selected.iloc[0]["latest_close"], 106.0)


if __name__ == "__main__":
    unittest.main()
