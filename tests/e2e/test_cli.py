import subprocess


def test_cli_script_help() -> None:
    result = subprocess.run(
        ["python", "-m", "imra.cli.main", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "IMRA-1" in result.stdout
