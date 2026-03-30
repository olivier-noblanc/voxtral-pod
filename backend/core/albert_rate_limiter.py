"""
Rate limiter et circuit breaker pour l'API Albert.
Permet de gérer les erreurs 429 et de basculer vers le mode cpu
en cas de dépassement de quota.
"""
import time
import asyncio
from typing import Optional
from collections import deque
import os
from threading import Lock
import threading

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
        self._fallback_task: Optional[object] = None
        
        # État du rate limiter
        self._consecutive_429 = 0
        self._last_429_time: Optional[float] = None
        self._in_mock_mode = False
        self._mock_mode_until: Optional[float] = None
        self._last_request_time: Optional[float] = None
        
        # Historique des codes de réponse pour le circuit breaker
        self._response_history = deque(maxlen=20)
        
        # Vérification de la clé API
        self._has_valid_api_key = bool(os.getenv("ALBERT_API_KEY"))
        
        # Lock pour les appels concurrents
        self._lock = Lock()
        
    def should_use_cpu_fallback_mode(self) -> bool:
        """Détermine si on doit activer le fallback CPU après dépassement du quota."""
        # Pas de clé API valide → on ne bascule jamais (on reste en mode mock pour les tests)
        if not self._has_valid_api_key:
            return False

        # Si le nombre de 429 consécutifs dépasse la limite, on active le fallback CPU
        if self._consecutive_429 >= self.max_429_count:
            from backend import state as backend_state
            backend_state.set_current_model("cpu")
            print(f"[*] {self._consecutive_429} 429 consécutifs – bascule vers fallback CPU")

            # Planifie le retour au modèle Albert après la durée de fallback actuelle
            def _revert():
                backend_state.set_current_model("albert")
                print(f"[RATELIMITER] Retour au modèle Albert après {self.current_fallback_duration}s")
                # Réinitialiser le compteur et la durée de fallback
                self._consecutive_429 = 0
                self.current_fallback_duration = self.base_fallback_duration

            # Annule tout timer de revert précédent
            if isinstance(self._fallback_task, threading.Timer):
                self._fallback_task.cancel()
            self._fallback_task = threading.Timer(self.current_fallback_duration, _revert)
            self._fallback_task.start()

            # Double la durée de fallback pour le prochain basculement (exponential backoff)
            self.current_fallback_duration = min(self.current_fallback_duration * 2, 24 * 3600)  # cap à 24 h

            return True

        return False
        
    def can_make_request(self) -> bool:
        """Vérifie si on peut faire une nouvelle requête selon le rate limit."""
        with self._lock:
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
                    print(f"[RATELIMITER] Réinitialisation du compteur 429 et remise à zéro du fallback.")
                    
    def get_retry_delay(self, attempt: int) -> float:
        """Calcule le délai de retry exponentiel."""
        if attempt < 3:
            return 2 ** attempt  # 1s, 2s, 4s
        return 30  # Maximum 30s
        
    def is_in_mock_mode(self) -> bool:
        """Vérifie si on est actuellement en mode mock."""
        return self._in_mock_mode
        
    def get_status_info(self) -> dict:
        """Retourne les informations d'état du rate limiter."""
        return {
            "in_mock_mode": self._in_mock_mode,
            "consecutive_429": self._consecutive_429,
            "has_valid_api_key": self._has_valid_api_key,
            "mock_mode_until": self._mock_mode_until,
            "last_429_time": self._last_429_time,
            "last_request_time": self._last_request_time
        }


# Instance globale du rate limiter
albert_rate_limiter = AlbertRateLimiter()
