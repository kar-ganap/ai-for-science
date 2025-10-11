"""
Integration Module: Active Generative Discovery

Tight coupling of VAE generation with Economic Active Learning.
"""

from .active_generative_discovery import (
    ActiveGenerativeDiscovery,
    create_al_candidate_pool
)

__all__ = [
    'ActiveGenerativeDiscovery',
    'create_al_candidate_pool'
]
