from src.utils.log_utils import flatten


class TestFlatten:
    def test_flatten_empty(self):
        empty = {}
        assert flatten(empty) == empty

    def test_flatten_simple(self):
        single_level = {"a": 1, "b": 2}
        assert flatten(single_level) == single_level

    def test_flatten_nested(self):
        nested = {"a": {"b": {"c": 1}}}
        assert flatten(nested) == {"a.b.c": 1}
