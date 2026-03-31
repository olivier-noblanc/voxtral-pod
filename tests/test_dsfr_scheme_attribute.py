"""
Test pour garantir que tous les templates HTML incluent l'attribut data-fr-scheme
Cela prévient la régression du problème de style DSFR
"""

import pytest


def test_all_templates_have_dsfr_scheme():
    """Test que tous les templates incluent l'attribut data-fr-scheme"""
    templates_to_check = [
        'backend/templates/index.html',
        'backend/templates/view.html', 
        'backend/templates/postprocess.html',
        'backend/templates/diarization_view.html'
    ]
    
    for template_path in templates_to_check:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Vérifier que l'attribut data-fr-scheme est présent dans le template
        assert 'data-fr-scheme="dark"' in content, f"Le template {template_path} doit inclure data-fr-scheme='dark'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
