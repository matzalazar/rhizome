"""
CLI bootstrap.

Loads .env before pydantic-settings reads environment variables, then
delegates entirely to cli/commands.py where all command definitions live.
"""

from dotenv import load_dotenv

load_dotenv()

from rhizome.cli.commands import app  # noqa: E402 — must come after load_dotenv()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
