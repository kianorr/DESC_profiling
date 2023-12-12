"""Use nox to run tests."""
import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests."""
    session.install("pytest>=7.0.0")
    session.run("pytest", *session.posargs)
