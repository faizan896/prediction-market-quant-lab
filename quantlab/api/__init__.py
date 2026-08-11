"""HTTP layer. Import `app` lazily so the core package never needs FastAPI."""

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from quantlab.api.main import app

        return app
    raise AttributeError(name)
