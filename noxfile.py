"""Use nox to run tests."""
import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run the unit and regular tests. Testing."""
    session.install("pytest")
    session.run("pytest", *session.posargs)
