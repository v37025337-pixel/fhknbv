import unittest

from digital_mind_core.github_sidecar import GitHubRepositorySidecar, RepositoryBinding


CURRENT = "7e968dea1f4286c8a1f36d3815a1cb014e8527c5"


def fake_head(repo, branch):
    if repo != "v37025337-pixel/fhknbv":
        raise AssertionError(repo)
    if branch != "main":
        raise AssertionError(branch)
    return {
        "sha": CURRENT,
        "commit": {"message": "Connect UnifiedKernel GitHub sidecar"},
        "html_url": f"https://github.com/{repo}/commit/{CURRENT}",
    }


class GitHubSidecarTests(unittest.TestCase):
    def test_synced_binding(self):
        sidecar = GitHubRepositorySidecar(bound_head=CURRENT)
        result = sidecar.verify(fake_head)
        self.assertEqual(result["status"], "SYNCED")
        self.assertEqual(result["remote_head"], CURRENT)
        self.assertEqual(result["transport"], "host_callback")
        self.assertFalse(sidecar.snapshot()["write_credentials_held"])

    def test_drift_detected(self):
        sidecar = GitHubRepositorySidecar(bound_head="0" * 40)
        self.assertEqual(sidecar.verify(fake_head)["status"], "DRIFTED")

    def test_unbound_repository(self):
        sidecar = GitHubRepositorySidecar()
        self.assertEqual(sidecar.verify(fake_head)["status"], "UNBOUND")

    def test_invalid_binding_rejected(self):
        with self.assertRaises(ValueError):
            RepositoryBinding("not a repo")


if __name__ == "__main__":
    unittest.main()
