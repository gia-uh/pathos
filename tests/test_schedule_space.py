import pytest
from pathos.spaces.schedule import ScheduleSpace


def test_constructor_accepts_entities_and_slots():
    s = ScheduleSpace(entities=["a", "b", "c"], slots=4)
    assert s._entities == ("a", "b", "c")
    assert s._slots == 4


def test_constructor_rejects_empty_entities():
    with pytest.raises(ValueError, match="entities"):
        ScheduleSpace(entities=[], slots=4)


def test_constructor_rejects_zero_or_negative_slots():
    with pytest.raises(ValueError, match="slots"):
        ScheduleSpace(entities=["a"], slots=0)
    with pytest.raises(ValueError, match="slots"):
        ScheduleSpace(entities=["a"], slots=-1)


def test_constructor_default_downstream_is_identity():
    s = ScheduleSpace(entities=["a", "b"], slots=2)
    assert list(s._downstream("a")) == ["a"]
    assert list(s._downstream("b")) == ["b"]


def test_constructor_custom_downstream():
    leaves = {"sub1": ["x", "y"], "sub2": ["z"]}
    s = ScheduleSpace(entities=["sub1", "sub2"], slots=2,
                       downstream=lambda e: leaves[e])
    assert list(s._downstream("sub1")) == ["x", "y"]
    assert list(s._downstream("sub2")) == ["z"]


def test_constructor_default_penalty_is_1000():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s._penalty == 1e3


def test_constructor_custom_penalty():
    s = ScheduleSpace(entities=["a"], slots=1, penalty=42.0)
    assert s._penalty == 42.0


def test_initial_state_is_empty_frozenset():
    s = ScheduleSpace(entities=["a", "b"], slots=3)
    assert s._initial == frozenset()


def test_capabilities_empty_before_any_decorator():
    s = ScheduleSpace(entities=["a"], slots=1)
    assert s.capabilities == set()
