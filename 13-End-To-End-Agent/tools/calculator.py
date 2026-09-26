from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    result = eval(expression)
    return str(result)
    """
    Calculate a mathematical expression.

    Use this tool when the user asks for arithmetic
    calculations such as addition, subtraction,
    multiplication, division, or powers.

    Args:
        expression: A mathematical expression to calculate.
    """

    try:

        result = eval(expression)

        return str(result)

    except Exception as error:

        return f"Calculation error: {error}"