"""Tests for the CancelToken cooperative-cancellation primitive."""
from __future__ import annotations

from pathos import Space
from pathos.core.cancel import CancelToken


def test_token_default_not_set():
    token = CancelToken()
    assert token.is_set() is False
    assert bool(token) is False


def test_token_request_cancel_sets_flag():
    token = CancelToken()
    token.request_cancel()
    assert token.is_set() is True
    assert bool(token) is True


def test_token_request_cancel_is_idempotent():
    token = CancelToken()
    token.request_cancel()
    token.request_cancel()
    assert token.is_set() is True


def test_space_has_cancel_token_by_default():
    space = Space()
    assert isinstance(space._cancel_token, CancelToken)
    assert space._cancel_requested() is False


def test_space_request_cancel_flips_flag():
    space = Space()
    space._request_cancel()
    assert space._cancel_requested() is True


def test_space_cancel_token_is_per_space_not_shared():
    s1 = Space()
    s2 = Space()
    s1._request_cancel()
    assert s2._cancel_requested() is False
