"""Legacy CLI shim that re-exports the Typer app."""

from imra.cli.main import app

if __name__ == "__main__":
    app()
