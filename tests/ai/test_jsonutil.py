import pytest

from torusplot.ai.jsonutil import (
    JSONExtractError,
    balanced_objects,
    contains_json_object,
    extract_json,
    strip_json_noise,
)


def test_plain_object():
    assert extract_json('{"plot_type": "bar", "data": {"table": "T"}}') == {"plot_type": "bar", "data": {"table": "T"}}
    assert extract_json('  \n{"a": 1}\n  ') == {"a": 1}
    assert extract_json('﻿{"a": 1}') == {"a": 1}


def test_fenced_block_inside_prose():
    text = 'Sure! Here is your figure:\n\n```json\n{"plot_type": "box", "n": [1, 2]}\n```\nLet me know if {this} helps.'
    assert extract_json(text) == {"plot_type": "box", "n": [1, 2]}


def test_json_fence_wins_over_other_fences():
    text = '```python\n{"not": "this"}\n```\nand\n```json\n{"yes": "this"}\n```'
    assert extract_json(text) == {"yes": "this"}
    assert extract_json('```\n{"plain": "fence"}\n```') == {"plain": "fence"}
    assert extract_json('``` JSON \r\n{"upper": 1}\r\n```') == {"upper": 1}


def test_first_balanced_object_in_prose():
    text = 'Use {curly} braces. The spec is {"a": {"b": "}"}, "c": "{"} and then {"second": 1}.'
    assert extract_json(text) == {"a": {"b": "}"}, "c": "{"}
    assert extract_json('prefix {"a": "quote \\" and brace }"} suffix') == {"a": 'quote " and brace }'}


def test_object_followed_by_prose():
    assert extract_json('{"a": 1}\nHope this helps!') == {"a": 1}


def test_trailing_commas_and_comments():
    text = """{
      // the plot type
      "plot_type": "bar",   // trailing comment
      "url": "http://example.com/a,}",   /* block
      comment */
      "layers": [ {"type": "mean_bar",}, ],
      "nested": {"a": 1, /* c */ },
    }"""
    assert extract_json(text) == {
        "plot_type": "bar",
        "url": "http://example.com/a,}",
        "layers": [{"type": "mean_bar"}],
        "nested": {"a": 1},
    }


def test_strip_json_noise_keeps_strings_intact():
    assert strip_json_noise('{"a": "x // y", "b": "1, ]"}') == '{"a": "x // y", "b": "1, ]"}'
    assert strip_json_noise('[1, 2, // c\n ]') == "[1, 2 \n ]"
    assert strip_json_noise('{"a": 1 /* never closed') == '{"a": 1 '
    assert strip_json_noise('{"a": 1} // tail') == '{"a": 1} '
    assert strip_json_noise('{"a": "esc \\" //x",}') == '{"a": "esc \\" //x"}'


def test_thinking_tags_are_ignored():
    text = '<think>I could answer {"wrong": true} but</think>\n{"right": true}'
    assert extract_json(text) == {"right": True}


def test_balanced_objects():
    assert balanced_objects('a {"x": 1} b {"y": {"z": 2}} c {unclosed') == ['{"x": 1}', '{"y": {"z": 2}}']
    assert balanced_objects("no braces") == []


@pytest.mark.parametrize("text,message", [
    ("", "empty"),
    ("   ", "empty"),
    ("Make the bars blue please.", "no '\\{'"),
    ("[1, 2, 3]", "no '\\{'"),
    ('{"a": 1', "No valid JSON object"),
    ("{not json at all}", "No valid JSON object"),
])
def test_errors(text, message):
    with pytest.raises(JSONExtractError, match=message):
        extract_json(text)


def test_non_string_input_and_top_level_array():
    with pytest.raises(JSONExtractError):
        extract_json(None)  # type: ignore[arg-type]
    # a fenced array is not an object; the object inside it is found by brace matching
    assert extract_json('```json\n[{"a": 1}]\n```') == {"a": 1}


def test_contains_json_object():
    assert contains_json_object('x {"a": 1} y')
    assert not contains_json_object("just words")
    assert not contains_json_object("{}")
