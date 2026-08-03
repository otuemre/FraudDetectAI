import sys


class CustomException(Exception):
    """Wraps an original exception with structured context for JSON logging.

    Usage:
        try:
            # code that may raise an exception
        except Exception as e:
            raise CustomException(e, sys) from e
    """

    def __init__(self, original_exception: Exception, error_detail: sys):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.error_type = type(original_exception).__name__
        self.message = str(original_exception)

        _, _, exc_tb = sys.exc_info()

        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.function_name = exc_tb.tb_frame.f_code.co_name
            self.line_number = exc_tb.tb_lineno
        else:
            self.file_name = None
            self.function_name = None
            self.line_number = None

    def to_dict(self):
        """Convert the exception details to a dictionary for JSON logging."""
        return {
            "error_type": self.error_type,
            "file_name": self.file_name,
            "function_name": self.function_name,
            "line_number": self.line_number,
            "message": self.message,
        }

    def __str__(self):
        return f"{self.error_type}: {self.message} (File: {self.file_name}, Function: {self.function_name}, Line: {self.line_number})"
