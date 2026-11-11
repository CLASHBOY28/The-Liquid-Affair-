import unittest

from inventory_matching import normalized_name, unique_item_names, unique_labels


class InventoryNameMatchingTests(unittest.TestCase):
    def test_normalized_name_ignores_case_and_outer_whitespace(self):
        self.assertEqual(normalized_name("  Glass "), normalized_name("gLaSs"))

    def test_unique_labels_preserves_first_display_spelling(self):
        self.assertEqual(unique_labels(["Glass", "glass", " Cups ", "cups"]), ["Glass", "Cups"])

    def test_item_choices_collapse_case_only_duplicates(self):
        records = [
            {"name": "Glass"},
            {"name": "glass"},
            {"name": "Mango Juice"},
            {"name": "mango juice"},
        ]
        self.assertEqual(unique_item_names(records), ["Glass", "Mango Juice"])


if __name__ == "__main__":
    unittest.main()

