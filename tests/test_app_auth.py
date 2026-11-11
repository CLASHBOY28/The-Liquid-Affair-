import unittest

from app_auth import load_auth_users


class AuthConfigurationTests(unittest.TestCase):
    def test_accepts_two_distinct_users(self):
        users = load_auth_users({
            "APP_USERNAME": "owner",
            "APP_PASSWORD": "owner-secret",
            "APP_USERNAME_2": "staff",
            "APP_PASSWORD_2": "staff-secret",
        })
        self.assertEqual(users, [("owner", "owner-secret"), ("staff", "staff-secret")])

    def test_second_user_is_optional_until_configured(self):
        self.assertEqual(load_auth_users({
            "APP_USERNAME": "owner", "APP_PASSWORD": "owner-secret"
        }), [("owner", "owner-secret")])

    def test_rejects_incomplete_user_pairs(self):
        for environment in (
            {"APP_USERNAME": "owner"},
            {"APP_USERNAME": "owner", "APP_PASSWORD": "secret", "APP_USERNAME_2": "staff"},
        ):
            with self.subTest(environment=environment):
                with self.assertRaises(ValueError):
                    load_auth_users(environment)

    def test_rejects_duplicate_usernames_case_insensitively(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            load_auth_users({
                "APP_USERNAME": "Owner", "APP_PASSWORD": "one",
                "APP_USERNAME_2": "owner", "APP_PASSWORD_2": "two",
            })


if __name__ == "__main__":
    unittest.main()

