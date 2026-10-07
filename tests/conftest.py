import pytest

from helpers import build_bundle


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    return build_bundle(tmp_path_factory.mktemp("bundle") / "artifacts")


@pytest.fixture(scope="session")
def classifier(artifact_dir):
    from sentiment import Classifier

    return Classifier(artifact_dir)


@pytest.fixture(scope="session")
def market(artifact_dir):
    from market import Market

    return Market(artifact_dir)


@pytest.fixture(scope="session")
def app(artifact_dir):
    from server import create_app

    return create_app(artifact_dir)


@pytest.fixture(scope="session")
def client(app):
    return app.test_client()
