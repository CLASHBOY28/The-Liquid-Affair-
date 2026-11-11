"""Credential configuration for the two named app users."""


def load_auth_users(environment):
    """Read one required account and one optional second account from env vars."""
    users = []
    for suffix in ("", "_2"):
        username_key = f"APP_USERNAME{suffix}"
        password_key = f"APP_PASSWORD{suffix}"
        username = (environment.get(username_key) or "").strip()
        password = environment.get(password_key) or ""

        if bool(username) != bool(password):
            raise ValueError(f"Set both {username_key} and {password_key}, or leave both empty.")
        if suffix == "" and not username:
            raise ValueError("Set APP_USERNAME and APP_PASSWORD before starting the app.")
        if not username:
            continue
        if any(existing.casefold() == username.casefold() for existing, _ in users):
            raise ValueError("The two app usernames must be different.")
        if any(ord(character) < 32 for character in username):
            raise ValueError("Usernames cannot contain control characters.")
        users.append((username, password))
    return users

