import unittest

from event_completion import aggregate_event_allocations, build_completion_rows


class EventCompletionTests(unittest.TestCase):
    def test_aggregates_duplicate_item_rows_in_first_seen_order(self):
        allocations = aggregate_event_allocations(
            ["Unit", "Item", "Quantity"],
            [["bottles", "Gin", 3], ["bottles", " gin ", 2], ["pcs", "Ice", 4]],
        )
        self.assertEqual(allocations, [["Gin", 5], ["Ice", 4]])

    def test_rejects_quantity_without_an_item(self):
        with self.assertRaisesRegex(ValueError, "Choose an inventory item"):
            aggregate_event_allocations(["Item", "Quantity"], [["", 2]])

    def test_computes_used_from_returned_and_keeps_every_allocation(self):
        rows = build_completion_rows(
            [["Gin", 8], ["Ice", 4]],
            [["Gin", 8, 3], ["Ice", 4, 0]],
        )
        self.assertEqual(rows, [["Gin", 8, 5, 3], ["Ice", 4, 4, 0]])

    def test_requires_a_return_quantity_for_each_item(self):
        with self.assertRaisesRegex(ValueError, "every allocated item"):
            build_completion_rows([["Gin", 8], ["Ice", 4]], [["Gin", 8, 3]])

    def test_rejects_return_greater_than_allocated(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed allocated"):
            build_completion_rows([["Gin", 8]], [["Gin", 8, 9]])

    def test_rejects_editing_read_only_item_and_allocation(self):
        for row in (["Vodka", 8, 2], ["Gin", 9, 2]):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "cannot be changed"):
                    build_completion_rows([["Gin", 8]], [row])

    def test_rejects_negative_or_fractional_returns(self):
        for value in (-1, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "whole-number"):
                    build_completion_rows([["Gin", 8]], [["Gin", 8, value]])

    def test_blank_return_requires_an_explicit_zero(self):
        with self.assertRaisesRegex(ValueError, "enter 0 if none"):
            build_completion_rows([["Gin", 8]], [["Gin", 8, None]])


if __name__ == "__main__":
    unittest.main()
