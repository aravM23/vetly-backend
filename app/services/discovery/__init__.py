"""Automated creator-discovery pipeline for Club Stanley."""
from app.services.discovery.runner import run_discovery, promote_candidate

__all__ = ["run_discovery", "promote_candidate"]
