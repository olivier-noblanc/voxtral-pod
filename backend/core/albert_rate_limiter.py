"""
Rate limiter et circuit breaker pour l'API Albert.
Permet de gérer les erreurs 429 et de basculer vers le mode cpu
en cas de dépassement de quota.
"""
import logging
import os
import threading
import time
from collections import deque
from threading import Lock
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Ensure NNPACK is disabled for rate limiter operations
os.environ.setdefault("DISABLE_NNPACK", "1")


class AlbertRateLimiter:
    """Gère les appels à l'API Albert avec rate limiting et circuit breaker."""
    
    def __init__(self):
        # Configuration
        self.max_429_count = 1  # Basculer dès la première 429
        self.reset_timeout = 900  # Temps de reset en secondes (15 minutes) - comme demandé
        self.min_interval_seconds = 1.0  # Intervalle minimal entre les requêtes
        # Fallback configuration
        self.base_fallback_duration = 900  # 15 minutes (en secondes)
        self.current_fallback_duration = self.base_fallback_duration
        self._fallback_task: Optional[threading.Timer] = None
        
        # État du rate limiter
        self._consecutive_429 = 0
        self._last_429_time: Optional[float] = None
        self._in_mock_mode = False
        self._mock_mode_until: Optional[float] = None
        self._last_request_time: Optional[float] = None
        
        # Historique des codes de réponse pour le circuit breaker
        self._response_history = deque(maxlen=20)
        
        # Vérification de la clé API
        from backend.config import get_albert_api_key
        self.albert_api_key = get_albert_api_key()
        self._has_valid_api_key = bool(self.albert_api_key)
        self.albert_base_url = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
        
        # Quotas (1000 est la valeur par défaut du user, on tentera de l'actualiser via /v1/me/info)
        self._quota_limit: int = 1000
        self._quota_usage: int = 0
        self._quota_asr_usage: int = 0
        self._quota_llm_usage: int = 0
        self._last_quota_update: float = 0.0
        self._quota_refresh_interval = 600  # 10 minutes
        
        # Lock pour les appels concurrents
        self._lock = Lock()
        # Indique si le fallback CPU est actuellement actif
        self._fallback_active: bool = False
        # Timestamp (epoch) jusqu’à lequel le fallback doit rester actif
        self._fallback_until: float = 0.0
        
    def should_use_cpu_fallback_mode(self) -> bool:
        """Détermine si on doit activer le fallback CPU après dépassement du quota."""
        # Si aucune clé API valide, on autorise le fallback en mode test (TESTING=1) ; sinon on ne bascule pas.
        if not self._has_valid_api_key and os.getenv("TESTING") != "1":
            return False

        # Si le nombre de 429 consécutifs dépasse la limite, on active le fallback CPU
        if self._consecutive_429 >= self.max_429_count:
            # Si le fallback est déjà actif, on ne recrée pas le timer
            if not self._fallback_active:
                from backend import state as backend_state
                backend_state.set_current_model("cpu")
                print(f"[*] {self._consecutive_429} 429 consécutifs – bascule vers fallback CPU")
                self._fallback_active = True
                self._fallback_until = time.time() + self.current_fallback_duration

                # Annule tout timer de revert précédent
                if isinstance(self._fallback_task, threading.Timer):
                    self._fallback_task.cancel()
                self._fallback_task = threading.Timer(self.current_fallback_duration, self._revert)
                self._fallback_task.start()

                # Double la durée de fallback pour le prochain basculement (exponential backoff)
                self.current_fallback_duration = min(self.current_fallback_duration * 2, 24 * 3600)  # cap à 24 h
            return True

        return False
        
    def _revert(self):
        """Fonction de revert pour revenir au modèle Albert après le fallback."""
        from backend import state as backend_state
        backend_state.set_current_model("albert")
        print(f"[RATELIMITER] Retour au modèle Albert après {self.current_fallback_duration}s")
        # Réinitialiser le compteur et la durée de fallback
        self._consecutive_429 = 0
        self.current_fallback_duration = self.base_fallback_duration
        self._fallback_active = False
        self._fallback_until = 0.0
        
    def can_make_request(self) -> bool:
        """Vérifie si on peut faire une nouvelle requête selon le rate limit."""
        with self._lock:
            # Bloquer les requêtes tant que le fallback CPU est actif
            if self._fallback_active and time.time() < self._fallback_until:
                remaining = int(self._fallback_until - time.time())
                print(f"[RATELIMITER] Fallback CPU actif – aucune requête API pendant encore {remaining}s")
                return False

            if self._last_request_time is not None:
                elapsed = time.time() - self._last_request_time
                return elapsed >= self.min_interval_seconds
            return True
            
    def record_request(self):
        """Enregistre qu'une requête a été effectuée."""
        with self._lock:
            self._last_request_time = time.time()
            
    def handle_response(self, status_code: int):
        """Traite une réponse de l'API Albert."""
        with self._lock:
            self._response_history.append(status_code)
            
            if status_code == 429:
                self._consecutive_429 += 1
                self._last_429_time = time.time()
                print(f"[RATELIMITER] 429 reçu. {self._consecutive_429} 429 consécutifs.")
            else:
                # Réinitialiser le compteur de 429 si ce n'est pas un 429
                if status_code != 429 and self._consecutive_429 > 0:
                    self._consecutive_429 = 0
                    self.current_fallback_duration = self.base_fallback_duration
                    print("[RATELIMITER] Réinitialisation du compteur 429 et remise à zéro du fallback.")
                    
    def get_retry_delay(self, attempt: int) -> float:
        """Calcule le délai de retry exponentiel."""
        if attempt < 3:
            return 2 ** attempt  # 1s, 2s, 4s
        return 30  # Maximum 30s
        
    def is_in_mock_mode(self) -> bool:
        """Vérifie si on est actuellement en mode mock."""
        return self._in_mock_mode
        
    def get_status_info(self) -> dict:
        """Retourne les informations d'état du rate limiter, y compris le statut de fallback."""
        return {
            "in_mock_mode": self._in_mock_mode,
            "consecutive_429": self._consecutive_429,
            "has_valid_api_key": self._has_valid_api_key,
            "mock_mode_until": self._mock_mode_until,
            "last_429_time": self._last_429_time,
            "last_request_time": self._last_request_time,
            "fallback_active": self._fallback_active,
            "fallback_until": self._fallback_until,
            "quota_limit": self._quota_limit,
            "quota_usage": self._quota_usage,
            "quota_asr_usage": self._quota_asr_usage,
            "quota_llm_usage": self._quota_llm_usage,
            "last_quota_update": self._last_quota_update
        }

    def update_quota_info(self):
        """Récupère l'usage et les limites du compte Albert."""
        if not self.albert_api_key:
            return

        # Éviter de rafraîchir trop souvent
        if time.time() - self._last_quota_update < self._quota_refresh_interval:
            return

        try:
            headers = {"Authorization": f"Bearer {self.albert_api_key}"}
            # 1. On tente d'abord de récupérer l'usage global (cumulé)
            usage_url = f"{self.albert_base_url}/me/usage"
            # On prend un nombre important de records pour l'usage récent
            r_usage = requests.get(f"{usage_url}?limit=100", headers=headers, timeout=5)
            if r_usage.status_code == 200:
                data = r_usage.json()
                if "data" in data and isinstance(data["data"], list):
                    total_global = 0
                    total_asr = 0
                    total_llm = 0
                    for item in data["data"]:
                        count = item.get("requests", 1)  # Si pas de requests, on compte 1 par record
                        total_global += count
                        
                        model = (item.get("model") or "").lower()
                        endpoint = (item.get("endpoint") or "").lower()
                        
                        # Classification ASR
                        if "whisper" in model or "/audio/" in endpoint:
                            total_asr += count
                        # Classification LLM
                        elif "chat" in endpoint or "completion" in endpoint:
                            total_llm += count
                            
                    self._quota_usage = total_global
                    self._quota_asr_usage = total_asr
                    self._quota_llm_usage = total_llm
            
            # 2. On tente de récupérer les infos de limite (optionnel)
            info_url = f"{self.albert_base_url}/me/info"
            resp_info = requests.get(info_url, headers=headers, timeout=5)
            if resp_info.status_code == 200:
                info_data = resp_info.json()
                if "limit" in info_data:
                    self._quota_limit = info_data["limit"]
                elif "quota" in info_data:
                    self._quota_limit = info_data["quota"]

            self._last_quota_update = time.time()
            logger.info(f"[RATELIMITER] Quota Albert (ASR): {self._quota_asr_usage}/{self._quota_limit} (Total: {self._quota_usage})")
            
        except Exception:
            # Silent fail for quota update to avoid breaking main flow
            pass


# Instance globale du rate limiter
albert_rate_limiter = AlbertRateLimiter()
