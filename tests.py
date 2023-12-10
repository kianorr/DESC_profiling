import nox


@nox.session
def tests(session):
    session.install("pytest>=7.0.0", "uncertainties")
    session.run("pytest")
