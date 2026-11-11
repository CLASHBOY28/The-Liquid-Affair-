"""Provider-independent capabilities shared by The Liquid Affair chatbot."""

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo


def calculate_math(expression: str) -> str:
    """Evaluate a restricted arithmetic expression without eval/exec."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 240:
        raise ValueError("Enter a short arithmetic expression.")
    expression = expression.strip().replace(",", "")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("I couldn't understand that math expression.") from exc

    binary_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def visit(node, depth=0):
        if depth > 20:
            raise ValueError("Expression is too complex.")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent is too large.")
            value = binary_ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            value = unary_ops[type(node.op)](visit(node.operand, depth + 1))
        else:
            raise ValueError("Only numbers, parentheses, and +, -, *, /, //, %, and ** are supported.")
        try:
            finite = math.isfinite(float(value))
        except OverflowError as exc:
            raise ValueError("The result is outside the supported range.") from exc
        if not finite or abs(value) > 1e100:
            raise ValueError("The result is outside the supported range.")
        return value

    try:
        result = visit(tree)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero is not allowed.") from exc
    except OverflowError as exc:
        raise ValueError("The result is outside the supported range.") from exc
    return f"{result:.12g}" if isinstance(result, float) else str(result)


def current_india_time_context() -> str:
    """Return fresh server time for every chat request, explicitly in India time."""
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return (
        "CURRENT DATE AND TIME (live server clock; refreshes every request): "
        + now.strftime("%A, %d %B %Y, %I:%M:%S %p IST (Asia/Kolkata)")
        + ". Use this as the source of truth for questions about today, now, "
          "the current time, or date/time calculations. Do not guess or reuse "
          "a time from earlier conversation turns."
    )

