import unittest

from audit_identity import pack_actor_note, unpack_actor_note


class AuditIdentityTests(unittest.TestCase):
    def test_round_trips_actor_and_note(self):
        packed = pack_actor_note("staff@example.com", "delta=-4")
        self.assertEqual(unpack_actor_note(packed), ("staff@example.com", "delta=-4"))

    def test_real_actor_column_wins_but_unpacks_legacy_packed_rows(self):
        packed = pack_actor_note("old-user", "table edit")
        self.assertEqual(unpack_actor_note(packed, actor="current-user"),
                         ("current-user", "table edit"))

    def test_old_unattributed_records_are_identified_as_legacy(self):
        self.assertEqual(unpack_actor_note("Dashboard"), ("Legacy", "Dashboard"))


if __name__ == "__main__":
    unittest.main()

