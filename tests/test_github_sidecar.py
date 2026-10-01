from digital_mind_core.github_sidecar import GitHubRepositorySidecar, RepositoryBinding


CURRENT = "6da405bd70b45d3f5d93799452a56441ec8a9c20"


def fake_head(repo, branch):
    assert repo == "v37025337-pixel/fhknbv"
    assert branch == "main"
    return {
        "sha": CURRENT,
        "commit": {"message": "Add unified DIGITAL_MIND module v0.2.1 with Codespaces and verification"},
        "html_url": f"https://github.com/{repo}/commit/{CURRENT}",
    }


def test_synced_binding():
    sidecar = GitHubRepositorySidecar(bound_head=CURRENT)
    result = sidecar.verify(fake_head)
    assert result["status"] == "SYNCED"
    assert result["remote_head"] == CURRENT
    assert result["transport"] == "host_callback"
    assert sidecar.snapshot()["write_credentials_held"] is False


def test_drift_detected():
    old = "0" * 40
    sidecar = GitHubRepositorySidecar(bound_head=old)
    assert sidecar.verify(fake_head)["status"] == "DRIFTED"


def test_unbound_repository():
    sidecar = GitHubRepositorySidecar()
    assert sidecar.verify(fake_head)["status"] == "UNBOUND"


def test_invalid_binding_rejected():
    try:
        RepositoryBinding("not a repo")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid repository was accepted")
