"""Use nox to run tests."""
import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests. Testing."""
    session.install("--upgrade", "pip")
    session.install("pytest", "numpy")
    session.run("pytest", *session.posargs)
