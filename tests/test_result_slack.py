from pathos.core.result import SearchResult


def test_searchresult_has_optional_slack_field_defaulting_to_none():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="Dummy", nodes_expanded=0, elapsed=0.0, found=True,
    )
    assert r.slack is None


def test_searchresult_slack_field_can_be_set():
    r = SearchResult(
        solution="x", path=None, cost=1.0,
        algorithm="Dummy", nodes_expanded=0, elapsed=0.0, found=True,
        slack=[1.5, 2.0, -0.5],
    )
    assert r.slack == [1.5, 2.0, -0.5]


def test_searchresult_not_found_factory_still_works():
    r = SearchResult.not_found("Dummy", 0, 0.0)
    assert r.slack is None
