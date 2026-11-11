"""Credential configuration for the two named app users."""


def load_auth_users(environment):
    """Load complete accounts, accepting the legacy Railway second-user keys.

    A partially configured account is skipped so a typo in the optional second
    account cannot take the whole service offline. At least one complete pair
    is still required, so authentication is never disabled as a fallback.
    """
    def read_pair(username_key, password_key):
        username = (environment.get(username_key) or "").strip()
        password = environment.get(password_key) or ""
        if not username and not password:
            return None, False
        if not username or not password:
            return None, True
        if any(ord(character) < 32 for character in username):
            raise ValueError("Usernames cannot contain control characters.")
        return (username, password), False

    primary, primary_incomplete = read_pair("APP_USERNAME", "APP_PASSWORD")
    canonical_second, canonical_incomplete = read_pair("APP_USERNAME_2", "APP_PASSWORD_2")
    legacy_second, legacy_incomplete = read_pair("APP_USERNAME2", "APP_PASSWORD2")

    if canonical_second and legacy_second and canonical_second != legacy_second:
        raise ValueError("Use only one second-user variable spelling, or make both pairs identical.")

    second = canonical_second or legacy_second
    users = [user for user in (primary, second) if user]

    if not users:
        raise ValueError("Configure at least one complete username/password pair before starting the app.")

    if len(users) == 2 and users[0][0].casefold() == users[1][0].casefold():
        raise ValueError("The two app usernames must be different.")

    if primary_incomplete or canonical_incomplete or legacy_incomplete:
        import logging

        if primary_incomplete:
            logging.warning("Ignoring incomplete login pair APP_USERNAME/APP_PASSWORD")
        if canonical_incomplete:
            logging.warning("Ignoring incomplete login pair APP_USERNAME_2/APP_PASSWORD_2")
        if legacy_incomplete:
            logging.warning("Ignoring incomplete login pair APP_USERNAME2/APP_PASSWORD2")

    return users

