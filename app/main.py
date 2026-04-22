from __future__ import annotations

import asyncio
from models import ErrorInput, ErrorExplanation
from decision import explain_error
from llm import get_llm_client


def print_explanation(result: ErrorExplanation) -> None:
    print("\n" + "=" * 50)
    print(f"  Error Type:  {result.error_type}")
    print("=" * 50)
    print(f"\nWhat happened:\n  {result.plain_explanation}")
    print(f"\nMost likely cause:\n  {result.likely_cause}")
    print("\nDebugging steps:")
    for step in result.debug_steps:
        print(f"  • {step}")
    print()


async def main() -> None:
    client = get_llm_client()

    sample = ErrorInput(
        error_message="TypeError: unsupported operand type(s) for +: 'int' and 'str'",
        code_snippet="""\
age = input("Enter your age: ")
years_until_100 = 100 + age
print(f"You have {years_until_100} years until you turn 100!")
""",
    )

    try:
        result = await explain_error(sample, client)
        print_explanation(result)
    except ValueError as e:
        print(f"\n[Mock client active — swap in a real LLMClient to see output]\n{e}")


if __name__ == "__main__":
    asyncio.run(main())