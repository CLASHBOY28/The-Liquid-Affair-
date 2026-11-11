import unittest
from datetime import datetime as RealDateTime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import chatbot_capabilities
from chatbot_capabilities import calculate_math, current_india_time_context


class CalculatorTests(unittest.TestCase):
    def test_arithmetic_and_percentage_examples(self):
        cases = {
            "(250*3)+18": "768",
            "500*18/100": "90",
            "(1200-150)/3": "350",
            "2**8": "256",
            "-7//2": "-4",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(calculate_math(expression), expected)

    def test_rejects_code_and_non_arithmetic_syntax(self):
        for expression in ("__import__('os').system('whoami')", "(1).__class__", "True", "[1, 2]"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculate_math(expression)

    def test_rejects_zero_division_and_out_of_range_results(self):
        for expression in ("5/0", "2**13", "1e200", "9" * 241):
            with self.subTest(expression=expression[:30]):
                with self.assertRaises(ValueError):
                    calculate_math(expression)

    def test_large_integer_float_conversion_is_reported_as_a_safe_error(self):
        with self.assertRaisesRegex(ValueError, "supported range"):
            calculate_math("9" * 200 + "**12")


class IndiaTimeTests(unittest.TestCase):
    def test_uses_fresh_india_local_time_in_context(self):
        instant = RealDateTime(2026, 11, 11, 3, 30, 0, tzinfo=timezone.utc)

        class FixedDateTime:
            @classmethod
            def now(cls, tz=None):
                return instant.astimezone(tz)

        with patch.object(chatbot_capabilities, "datetime", FixedDateTime):
            context = current_india_time_context()

        self.assertIn("Wednesday, 11 November 2026, 09:00:00 AM IST (Asia/Kolkata)", context)
        self.assertIn("live server clock; refreshes every request", context)

    def test_iana_timezone_is_india_standard_time(self):
        self.assertEqual(
            ZoneInfo("Asia/Kolkata").utcoffset(RealDateTime.now()),
            timedelta(hours=5, minutes=30),
        )


if __name__ == "__main__":
    unittest.main()

