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

    def test_accepts_legacy_railway_second_user_variable_names(self):
        users = load_auth_users({
            "APP_USERNAME": "owner", "APP_PASSWORD": "owner-secret",
            "APP_USERNAME2": "staff", "APP_PASSWORD2": "staff-secret",
        })
        self.assertEqual(users, [("owner", "owner-secret"), ("staff", "staff-secret")])

    def test_incomplete_primary_pair_does_not_take_down_complete_second_user(self):
        users = load_auth_users({
            "APP_USERNAME": "owner",
            "APP_USERNAME2": "staff", "APP_PASSWORD2": "staff-secret",
        })
        self.assertEqual(users, [("staff", "staff-secret")])

    def test_incomplete_optional_second_pair_is_ignored(self):
        users = load_auth_users({
            "APP_USERNAME": "owner", "APP_PASSWORD": "owner-secret",
            "APP_USERNAME_2": "staff",
        })
        self.assertEqual(users, [("owner", "owner-secret")])

    def test_rejects_configuration_without_any_complete_pair(self):
        for environment in (
            {},
            {"APP_USERNAME": "owner"},
            {"APP_USERNAME_2": "staff"},
        ):
            with self.subTest(environment=environment):
                with self.assertRaisesRegex(ValueError, "at least one complete"):
                    load_auth_users(environment)

    def test_rejects_duplicate_usernames_case_insensitively(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            load_auth_users({
                "APP_USERNAME": "Owner", "APP_PASSWORD": "one",
                "APP_USERNAME_2": "owner", "APP_PASSWORD_2": "two",
            })

    def test_rejects_conflicting_second_user_spellings(self):
        with self.assertRaisesRegex(ValueError, "only one second-user"):
            load_auth_users({
                "APP_USERNAME": "owner", "APP_PASSWORD": "owner-secret",
                "APP_USERNAME_2": "staff", "APP_PASSWORD_2": "staff-secret",
                "APP_USERNAME2": "other", "APP_PASSWORD2": "other-secret",
            })


if __name__ == "__main__":
    unittest.main()

