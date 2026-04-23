"""Smoke tests — verify the package imports and basic objects are constructable."""

import pytest

import polyvia


def test_version_string():
    assert isinstance(polyvia.__version__, str)
    assert polyvia.__version__


def test_public_exports():
    assert hasattr(polyvia, "Polyvia")
    assert hasattr(polyvia, "AsyncPolyvia")
    assert hasattr(polyvia, "MCPConfig")


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("POLYVIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        polyvia.Polyvia()


def test_client_accepts_explicit_key():
    client = polyvia.Polyvia(api_key="poly_test")
    assert client is not None


def test_mcp_config_shape():
    client = polyvia.Polyvia(api_key="poly_test")
    mcp = client.mcp

    anthropic = mcp.to_anthropic_mcp_server()
    assert anthropic["type"] == "url"
    assert "polyvia.ai/mcp" in anthropic["url"]
    assert "Authorization" in anthropic["headers"]

    openai_resp = mcp.to_openai_responses_tool()
    assert openai_resp["type"] == "mcp"
    assert "polyvia.ai/mcp" in openai_resp["server_url"]

    openai_agents = mcp.to_openai_mcp_server()
    assert "url" in openai_agents
    assert "headers" in openai_agents

    desktop = mcp.to_claude_desktop_config()
    assert desktop["type"] == "http"


def test_exception_hierarchy():
    assert issubclass(polyvia.AuthenticationError, polyvia.APIError)
    assert issubclass(polyvia.NotFoundError, polyvia.APIError)
    assert issubclass(polyvia.RateLimitError, polyvia.APIError)
    assert issubclass(polyvia.ForbiddenError, polyvia.APIError)
    assert issubclass(polyvia.IngestionError, polyvia.PolyviaError)
    assert issubclass(polyvia.IngestionTimeout, polyvia.PolyviaError)
