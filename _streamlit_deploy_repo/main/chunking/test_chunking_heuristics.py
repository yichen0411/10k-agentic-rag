"""Unit tests for 10-K chunking heuristics (cross-page tables, headings)."""

from __future__ import annotations

import unittest

from section_asset_extractor import (
    find_section_for_page,
    is_data_row,
    is_header_only_table,
    is_narrow_schedule_table,
    is_prose_not_table_header,
    is_table_header_band,
    is_table_line_item_row,
    line_index_by_offset,
    looks_like_header_row,
    looks_like_prose_table,
    pick_table_anchor_line,
    same_page_fragment_candidate,
    should_merge_tables,
    table_crop_padding,
    PageTableFindCache,
    words_to_row_cells,
)
from table_pipeline import regions_should_link
from subsection_table_filter import collect_table_regions, heading_inside_table_region
from rag_chunk_builder import chunk_with_header, clean_body_for_text_chunk, header_text
from toc_guided_section_probe import centered_heading_score, is_subsection_candidate


class SectionLookupTests(unittest.TestCase):
    def test_same_start_page_picks_earlier_item_above_footer_band(self):
        refs = [
            {
                "section_ref_id": "Part II::Item 7A",
                "start_pdf_page": 47,
                "end_pdf_page": 47,
                "start_offset": 100,
            },
            {
                "section_ref_id": "Part II::Item 8",
                "start_pdf_page": 47,
                "end_pdf_page": 82,
                "start_offset": 200,
            },
        ]
        sensitivity_table_y = 543.75
        footer_y = 750.0
        self.assertEqual(
            find_section_for_page(refs, 47, sensitivity_table_y)["section_ref_id"],
            "Part II::Item 7A",
        )
        self.assertEqual(
            find_section_for_page(refs, 47, footer_y)["section_ref_id"],
            "Part II::Item 8",
        )

    def test_page_top_continuation_prefers_spanning_section(self):
        refs = [
            {
                "section_ref_id": "Part II::Item 7A",
                "start_pdf_page": 47,
                "end_pdf_page": 47,
                "start_offset": 100,
            },
            {
                "section_ref_id": "Part II::Item 8",
                "start_pdf_page": 47,
                "end_pdf_page": 82,
                "start_offset": 200,
            },
        ]
        from section_asset_extractor import find_section_for_asset

        self.assertEqual(
            find_section_for_asset(refs, 48, 75.0)["section_ref_id"],
            "Part II::Item 8",
        )


class PageTableFindCacheTests(unittest.TestCase):
    def test_find_tables_called_once_per_page(self):
        calls: list[int] = []

        class FakePage:
            def __init__(self, number: int) -> None:
                self.number = number

            def find_tables(self):
                calls.append(self.number)
                return type("Found", (), {"tables": []})()

        cache = PageTableFindCache()
        for page_no in (0, 0, 1, 1, 1):
            cache.find_tables(FakePage(page_no))
        self.assertEqual(calls, [0, 1])


class ProseTableFilterTests(unittest.TestCase):
    def test_rejects_three_column_prose_paragraphs(self):
        rows = [
            [None, None, "Our people are critical for our continued success, so we work hard to create an environment"],
            ["where employees can have fulfilling careers and perform at a high level.", None, None],
            [None, "We offer industry-leading benefits and programs to take care of the diverse needs", None],
            ["of our employees and their families.", None, None],
        ]
        self.assertTrue(looks_like_prose_table(rows))

    def test_keeps_numeric_repurchase_table(self):
        rows = [
            ["October 1 - 31", None, "841", None, "13,305", None, "$", "253.20"],
            ["November 1 - 30", None, "209", None, "5,885", None, "$", "286.00"],
            ["December 1 - 31", None, "0", None, "0", None, "$", "0.00"],
            ["Total", None, "1,050", None, "19,190", None, None, None],
        ]
        self.assertFalse(looks_like_prose_table(rows))

    def test_rejects_section_heading_label_split(self):
        rows = [["ITEM 1.", "BUSINESS"]]
        self.assertTrue(looks_like_prose_table(rows))
        rows = [["ITEM 2.", "PROPERTIES"]]
        self.assertTrue(looks_like_prose_table(rows))

    def test_rejects_wide_sparse_holder_style_prose(self):
        rows = [
            ["As of", "January 1, 2024", ", there were approximately", None, "1,234 and 567 holders", None, "of our Class A stock", None],
            ["respectively as detailed in the following discussion.", None, None, None, None, None, None, None],
            ["Many shares are held by brokers and institutions on behalf", None, None, None, None, None, None, None],
            ["of stockholders. As of January 1, 2024, ", None, None, None, None, "89", "ho", "lders of Class B stock"],
        ]
        self.assertTrue(looks_like_prose_table(rows))

    def test_rejects_short_wrapped_prose_grid(self):
        rows = [
            ["As of", "December 31, 2025", ", we had $126.8 b"],
            ["sh equivalents and marketable securities a", None, None],
        ]
        self.assertTrue(looks_like_prose_table(rows))

    def test_rejects_two_row_wrapped_prose_with_sparse_cells(self):
        rows = [
            ["The company maintains a repurchase program", "authorized by the board of directors in April", None],
            ["for its common shares and may repurchase stock", None, "from time to time in open market transactions"],
        ]
        self.assertTrue(looks_like_prose_table(rows))

    def test_rejects_two_cell_sparse_prose_wrap(self):
        rows = [
            [None, "We are a company of curious, talented, and passionate people. We embrace collaboration and creativity, and", None],
            ["encourage the iteration of ideas to address complex challenges in technology and society.", None, None],
        ]
        self.assertTrue(looks_like_prose_table(rows))

    def test_rejects_short_long_rows_in_sparse_grid(self):
        rows = [
            ["•", "Our products: We are empowering people with information about the environmental impacts of their choices."],
            ["•", "Our operations: We are working to drive sustainability and efficiency across our operations and value chain."],
            [None, "Through our products, we have an aspiration to help individuals, cities, and other partners collectively reduce one", None],
            ["gigaton of their carbon equivalent emissions annually by 2030.", None, None],
        ]
        self.assertTrue(looks_like_prose_table(rows))

    def test_keeps_cash_flow_line_item_table(self):
        rows = [
            ["Net cash provided by operating activities", "$", "125,299", "$", "164,713"],
            ["Net cash used in investing activities", "$", "(45,536)", "$", "(120,291)"],
            ["Net cash used in financing activities", "$", "(79,733)", "$", "(37,388)"],
        ]
        self.assertFalse(looks_like_prose_table(rows))


class TableCropPaddingTests(unittest.TestCase):
    def test_cross_page_table_group_gets_generous_top_pad(self):
        table = {
            "asset_type": "table_group",
            "header_only": False,
            "bbox_by_page": [{"page": 48, "bbox": [42, 657, 569, 711]}, {"page": 49, "bbox": [42, 54, 569, 108]}],
        }
        top, bottom = table_crop_padding(table, 0, table["bbox_by_page"][0]["bbox"], page_height=792.0)
        self.assertGreaterEqual(top, 96.0)
        self.assertGreater(top, 4.0)

    def test_header_only_band_stays_tight(self):
        table = {
            "header_only": True,
            "bbox_by_page": [{"page": 48, "bbox": [42, 750, 569, 780]}, {"page": 49, "bbox": [42, 54, 569, 200]}],
        }
        top, _ = table_crop_padding(table, 0, table["bbox_by_page"][0]["bbox"], page_height=792.0)
        self.assertEqual(top, 4.0)


class TableAnchorLineTests(unittest.TestCase):
    def test_pick_table_anchor_falls_back_to_table_line(self):
        caption_lines: list[dict] = []
        table_line = {"offset": 111204, "line": "75.4%", "y0": 554.0}
        anchor = pick_table_anchor_line(caption_lines, [table_line], y0=560.0)
        self.assertIs(anchor, table_line)
        self.assertIsNone(line_index_by_offset(caption_lines, anchor))

    def test_line_index_by_offset_matches_same_offset(self):
        lines = [{"offset": 10, "line": "A"}, {"offset": 20, "line": "B"}]
        self.assertEqual(line_index_by_offset(lines, {"offset": 20, "line": "copy"}), 1)


class HeadingDetectionTests(unittest.TestCase):
    def test_centered_all_caps_item5_heading(self):
        record = {
            "line": "SHARE REPURCHASES AND DIVIDENDS",
            "x0": 208.78,
            "x1": 403.21,
            "line_center": 305.99,
            "page_width": 612.0,
            "page_center": 306.0,
            "max_size": 10.0,
            "gap_before": 13.5,
            "gap_after": 9.0,
        }
        self.assertGreaterEqual(centered_heading_score(record), 4)
        self.assertTrue(is_subsection_candidate(record))

    def test_left_aligned_all_caps_with_gaps(self):
        record = {
            "line": "MARKET AND STOCKHOLDERS",
            "x0": 229.05,
            "x1": 382.94,
            "line_center": 305.99,
            "page_width": 612.0,
            "page_center": 306.0,
            "max_size": 10.0,
            "gap_before": 9.4,
            "gap_after": 9.0,
        }
        self.assertGreaterEqual(centered_heading_score(record), 4)
        self.assertTrue(is_subsection_candidate(record))

    def test_prose_line_not_subsection(self):
        record = {
            "line": "Our common stock is traded on the NASDAQ Stock Market under the symbol MSFT.",
            "x0": 42.0,
            "x1": 500.0,
            "line_center": 271.0,
            "page_width": 612.0,
            "page_center": 306.0,
            "max_size": 10.0,
            "gap_before": 4.0,
            "gap_after": 4.0,
        }
        self.assertFalse(is_subsection_candidate(record))

    def test_small_bold_salesforce_heading(self):
        record = {
            "line": "Business Benefits of Using Our Solutions",
            "x0": 39.3,
            "x1": 184.9,
            "line_center": 112.1,
            "page_width": 612.0,
            "page_center": 306.0,
            "max_size": 8.4,
            "is_bold": True,
            "is_italic": False,
            "gap_before": 14.0,
            "gap_after": 5.0,
        }
        self.assertTrue(is_subsection_candidate(record))

    def test_inline_italic_salesforce_heading(self):
        record = {
            "line": "Slack",
            "x0": 39.3,
            "x1": 61.6,
            "line_center": 50.4,
            "page_width": 612.0,
            "page_center": 306.0,
            "max_size": 8.4,
            "is_bold": False,
            "is_italic": True,
            "gap_before": 4.0,
            "gap_after": 0.0,
        }
        self.assertTrue(is_subsection_candidate(record))


class HeaderBandTests(unittest.TestCase):
    def test_dividend_header_band(self):
        rows = [
            ["Declaration Date", "Record Date", "Payment Date", "Per Share", "Amount"],
        ]
        self.assertTrue(is_table_header_band(rows))
        self.assertTrue(is_header_only_table(rows))

    def test_unearned_revenue_period_header_band(self):
        rows = [
            ["(In millions)"],
            ["Three Months Ending"],
        ]
        self.assertTrue(is_table_header_band(rows))

    def test_thereafter_and_total_rows(self):
        self.assertTrue(is_table_line_item_row(["Thereafter", "2,710"]))
        self.assertTrue(is_table_line_item_row(["Total", "$", "67,265"]))

    def test_narrow_schedule_table(self):
        table = {
            "col_count": 3,
            "raw_rows": [
                ["September 30, 2025", "$", "25,191"],
                ["Total", "$", "67,265"],
            ],
        }
        self.assertTrue(is_narrow_schedule_table(table))

    def test_prose_at_page_bottom_rejected(self):
        rows = [
            [
                "Our Board of Directors declared the following dividends during the fourth quarter of fiscal year 2025:",
            ],
        ]
        self.assertTrue(is_prose_not_table_header(rows[0][0]))
        self.assertFalse(is_table_header_band(rows))

    def test_dividend_data_row_not_header(self):
        row = ["June 10, 2025", "August 21, 2025", "September 11, 2025", "$", "0.83", "$", "6,170"]
        self.assertTrue(is_data_row(row))
        self.assertFalse(looks_like_header_row(row))

    def test_operating_income_word_row(self):
        cells = words_to_row_cells(
            [
                (49.0, "Operating"),
                (86.0, "Income"),
                (332.0, "$"),
                (376.0, "128,528"),
                (455.0, "$"),
                (490.0, "109,433"),
                (545.0, "17%"),
            ]
        )
        self.assertEqual(cells[0], "Operating Income")
        self.assertTrue(is_table_line_item_row(cells))
        self.assertTrue(is_data_row(cells))


class SegmentTableRescanTests(unittest.TestCase):
    def test_segment_subtotal_row_is_table_line_item(self):
        cells = ["Operating Income", "$", "69,773", "$", "59,661", "17%"]
        self.assertTrue(is_table_line_item_row(cells))

    def test_segment_heading_is_not_table_line_item(self):
        cells = ["Intelligent Cloud"]
        self.assertFalse(is_table_line_item_row(cells))


class CrossPageMergeTests(unittest.TestCase):
    class _Doc:
        def __getitem__(self, idx: int) -> "_Page":
            return self._Page()

        class _Page:
            rect = type("R", (), {"height": 792.0})()

    def test_header_only_page_bottom_merges_with_next_page_data(self):
        prev = {
            "page_end": 32,
            "bbox": [42.0, 735.0, 570.0, 749.0],
            "col_count": 5,
            "pending_merge": True,
            "header_only": True,
            "section_ref": {"section_ref_id": "Part II::Item 5"},
            "subsection_ref": {"subsection_ref_id": "Part II::Item 5::sub_002", "title": "SHARE REPURCHASES AND DIVIDENDS"},
            "raw_rows": [["Declaration Date", "Record Date", "Payment Date", "Per Share", "Amount"]],
        }
        nxt = {
            "page_start": 33,
            "bbox": [42.0, 27.75, 570.0, 40.36],
            "col_count": 17,
            "section_ref": {"section_ref_id": "Part II::Item 7"},
            "raw_rows": [["June 10, 2025", "August 21, 2025", "September 11, 2025", "$", "0.83", "$", "6,170"]],
        }
        ok, score, reasons = should_merge_tables(prev, nxt, self._Doc())
        self.assertTrue(ok, msg=f"score={score} reasons={reasons}")
        self.assertEqual(nxt["section_ref"]["section_ref_id"], "Part II::Item 5")
        self.assertEqual(nxt["subsection_ref"]["subsection_ref_id"], "Part II::Item 5::sub_002")

    def test_unrelated_tables_do_not_merge(self):
        prev = {
            "page_end": 10,
            "bbox": [42.0, 700.0, 570.0, 740.0],
            "col_count": 4,
            "header_only": False,
            "pending_merge": False,
            "section_ref": {"section_ref_id": "Part I::Item 1"},
            "raw_rows": [["Assets", "2025", "2024", "2023"]],
        }
        nxt = {
            "page_start": 11,
            "bbox": [42.0, 40.0, 570.0, 120.0],
            "col_count": 9,
            "section_ref": {"section_ref_id": "Part I::Item 1A"},
            "raw_rows": [["Revenue", "100", "200", "300"]],
        }
        ok, _, _ = should_merge_tables(prev, nxt, self._Doc())
        self.assertFalse(ok)


class TablePipelineSmokeTests(unittest.TestCase):
    def test_regions_should_link_same_page_only(self):
        prev = {
            "page_start": 43,
            "page_end": 43,
            "bbox": [42.0, 27.0, 570.0, 40.0],
            "col_count": 3,
            "section_ref": {"section_ref_id": "Part II::Item 7"},
            "subsection_ref": {"subsection_ref_id": "Part II::Item 7::sub_035"},
            "raw_rows": [["September 30, 2025", "$", "25,191"]],
        }
        nxt = {
            "page_start": 43,
            "page_end": 43,
            "bbox": [42.0, 50.0, 570.0, 62.0],
            "col_count": 3,
            "section_ref": {"section_ref_id": "Part II::Item 7"},
            "subsection_ref": {"subsection_ref_id": "Part II::Item 7::sub_035"},
            "raw_rows": [["December 31, 2025", "19,733"]],
        }
        self.assertTrue(regions_should_link(prev, nxt, CrossPageMergeTests._Doc()))


class SamePageFinancialStatementMergeTests(unittest.TestCase):
    def _balance_sheet_fragment(self, table_id: str, y0: float, y1: float, col_count: int) -> dict:
        return {
            "table_id": table_id,
            "page_start": 49,
            "page_end": 49,
            "bbox": [42.0, y0, 570.0, y1],
            "col_count": col_count,
            "section_ref": {"section_ref_id": "Part II::Item 8"},
            "subsection_ref": {"subsection_ref_id": "Part II::Item 8::sub_003"},
            "raw_rows": [["Assets", "1"]],
        }

    def test_balance_sheet_vertical_stack_merges_despite_column_mismatch(self):
        prev = self._balance_sheet_fragment("a", 138.0, 199.5, 9)
        nxt = self._balance_sheet_fragment("b", 210.75, 250.4, 3)
        self.assertTrue(same_page_fragment_candidate(prev, nxt))

    def test_balance_sheet_rescan_overlap_merges(self):
        prev = self._balance_sheet_fragment("a", 261.0, 362.9, 5)
        nxt = self._balance_sheet_fragment("b", 338.39, 569.25, 9)
        self.assertTrue(same_page_fragment_candidate(prev, nxt))


class SubsectionTableFilterTests(unittest.TestCase):
    def test_drops_balance_sheet_row_label_inside_table(self):
        regions = [{"page": 49, "bbox": [40.0, 600.0, 570.0, 740.0]}]
        assets_line = [42.0, 618.0, 120.0, 628.0]
        self.assertTrue(heading_inside_table_region(49, assets_line, regions))

    def test_keeps_statement_title_above_table(self):
        regions = [{"page": 49, "bbox": [40.0, 600.0, 570.0, 740.0]}]
        title_line = [260.0, 560.0, 350.0, 572.0]
        self.assertFalse(heading_inside_table_region(49, title_line, regions))

    def test_collect_table_regions_from_group(self):
        tables = [
            {
                "page_start": 42,
                "bbox_by_page": [{"page": 42, "bbox": [1, 2, 3, 4]}, {"page": 43, "bbox": [5, 6, 7, 8]}],
            }
        ]
        regions = collect_table_regions(tables)
        self.assertEqual(len(regions), 2)


class TableOnlySubsectionChunkTests(unittest.TestCase):
    def test_financial_statement_subsection_emits_table_marker_chunk(self):
        table_id = "table_group_013"
        body = (
            "INCOME STATEMENTS (In millions, except per share amounts) "
            "Year Ended June 30, 2025 2024 2023 Revenue: Product $ 63,946 $ 64,773"
        )
        tables_by_id = {
            table_id: {
                "table_id": table_id,
                "raw_rows": [
                    ["Year Ended June 30", "2025", "2024", "2023"],
                    ["Revenue", "281,724", "245,122", "211,915"],
                ],
            }
        }
        cleaned, anchors, refs = clean_body_for_text_chunk(body, "INCOME STATEMENTS", [table_id], tables_by_id)
        self.assertEqual(refs, [table_id])
        self.assertIn("[[TABLE:table_group_013]]", cleaned)
        header = header_text("Financial Statements", ["INCOME STATEMENTS"], part="Part II", item="Item 8")
        pieces = chunk_with_header(cleaned, header)
        self.assertEqual(len(pieces), 1)
        self.assertIn("INCOME STATEMENTS", pieces[0])
        self.assertIn("[[TABLE:table_group_013]]", pieces[0])
        self.assertEqual(len(anchors), 1)


if __name__ == "__main__":
    unittest.main()
