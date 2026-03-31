from unittest.mock import MagicMock, patch

import pytest

from backend.config import get_albert_api_key
from backend.core.albert_rate_limiter import AlbertRateLimiter


def test_get_albert_api_key_logic() -> None:
    """Vérifie la priorité de récupération de la clé API."""
    with patch("os.getenv", return_value="env_key"):
        assert get_albert_api_key() == "env_key"
        
    with patch("os.getenv", return_value=None):
        with patch("keyring.get_password", return_value="keyring_key"):
            assert get_albert_api_key() == "keyring_key"

@pytest.fixture
def rate_limiter():
    # Déclencher une nouvelle instance
    rl = AlbertRateLimiter()
    # On s'assure qu'on a une clé (via config ou mockée) pour que les tests de quota s'exécutent
    rl.albert_api_key = get_albert_api_key() or "mock_key_for_test"
    rl._has_valid_api_key = True
    return rl

def test_update_quota_info_success(rate_limiter):
    """Vérifie que le quota est correctement mis à jour depuis les réponses API."""
    
    mock_usage_resp = MagicMock()
    mock_usage_resp.status_code = 200
    mock_usage_resp.json.return_value = {
        "data": [
            {"requests": 50},
            {"requests": 25}
        ]
    }
    
    mock_info_resp = MagicMock()
    mock_info_resp.status_code = 200
    mock_info_resp.json.return_value = {"limit": 2000}

    # On mock requests.get
    with patch("requests.get") as mock_get:
        # Premier appel pour usage, second pour info
        mock_get.side_effect = [mock_usage_resp, mock_info_resp]
        
        rate_limiter.update_quota_info()
        
        info = rate_limiter.get_status_info()
        assert info["quota_usage"] == 75
        assert info["quota_limit"] == 2000
        assert info["last_quota_update"] > 0

def test_update_quota_info_throttle(rate_limiter):
    """Vérifie que l'intervalle de rafraîchissement (600s) est respecté."""
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": []}
    
    with patch("requests.get", return_value=mock_resp) as mock_get:
        # Première mise à jour
        rate_limiter.update_quota_info()
        assert mock_get.call_count >= 1
        
        last_update = rate_limiter._last_quota_update
        
        # Deuxième mise à jour immédiate
        rate_limiter.update_quota_info()
        assert rate_limiter._last_quota_update == last_update

def test_update_quota_info_error_handling(rate_limiter):
    """Vérifie que les erreurs API ne font pas planter le rate limiter."""
    
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    
    with patch("requests.get", return_value=mock_resp):
        # Ne doit pas lever d'exception
        rate_limiter.update_quota_info()
        
        info = rate_limiter.get_status_info()
        assert info["quota_usage"] == 0
        assert info["quota_limit"] == 1000
