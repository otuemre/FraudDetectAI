import sys

from src.exception import CustomException


def _raise_and_capture() -> CustomException:
    """
    Triggers a real exception inside a known function/line, then wraps it
    in CustomException — mirroring exactly how it's used in real code
    (raise CustomException(e, sys) from e inside an except block).
    """
    try:
        result = 1 / 0  # ZeroDivisionError
        return result
    except Exception as e:  # noqa: BLE001
        return CustomException(e, sys)


def test_custom_exception_captures_error_type():
    ce = _raise_and_capture()
    assert ce.error_type == "ZeroDivisionError"


def test_custom_exception_captures_message():
    ce = _raise_and_capture()
    assert "division by zero" in ce.message.lower()


def test_custom_exception_captures_function_name():
    ce = _raise_and_capture()
    assert ce.function_name == "_raise_and_capture"


def test_custom_exception_captures_file_name():
    ce = _raise_and_capture()
    assert ce.file_name.endswith("test_exception.py")


def test_custom_exception_captures_line_number():
    ce = _raise_and_capture()
    assert isinstance(ce.line_number, int)
    assert ce.line_number > 0


def test_to_dict_contains_all_expected_keys():
    ce = _raise_and_capture()
    result = ce.to_dict()

    expected_keys = {
        "error_type",
        "file_name",
        "function_name",
        "line_number",
        "message",
    }
    assert set(result.keys()) == expected_keys


def test_to_dict_values_match_attributes():
    ce = _raise_and_capture()
    result = ce.to_dict()

    assert result["error_type"] == ce.error_type
    assert result["file_name"] == ce.file_name
    assert result["function_name"] == ce.function_name
    assert result["line_number"] == ce.line_number
    assert result["message"] == ce.message


def test_str_representation_is_readable():
    ce = _raise_and_capture()
    text = str(ce)

    assert "ZeroDivisionError" in text
    assert "_raise_and_capture" in text
    assert "test_exception.py" in text


def test_custom_exception_preserves_original_exception_reference():
    ce = _raise_and_capture()
    assert isinstance(ce.original_exception, ZeroDivisionError)
