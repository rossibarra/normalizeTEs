from release_provenance import PROJECT_VERSION, software_provenance


def test_release_provenance_identifies_checkout():
    provenance = software_provenance()
    assert PROJECT_VERSION == "0.2.0"
    assert provenance["name"] == "normalizeTE"
    assert provenance["version"] == PROJECT_VERSION
    assert len(provenance["git_commit"]) == 40
    assert provenance["git_describe"]
    assert isinstance(provenance["git_dirty"], bool)
