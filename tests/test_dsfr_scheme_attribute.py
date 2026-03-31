"""
Test pour garantir que tous les templates HTML incluent l'attribut data-fr-scheme
et contiennent la structure DSFR complète
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


def test_index_template_has_complete_dsfr_structure():
    """Test que le template index.html contient la structure DSFR complète"""
    with open('backend/templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifier les classes DSFR essentielles
    essential_dsfr_classes = [
        'fr-header',
        'fr-container',
        'fr-btn',
        'fr-badge',
        'fr-grid-row',
        'fr-select-group'
    ]
    
    for cls in essential_dsfr_classes:
        assert cls in content, f"Le template index.html doit inclure la classe DSFR '{cls}'"
    
    # Vérifier la présence de structures HTML DSFR (avec gestion des espaces et retours à la ligne)
    essential_elements = [
        '<header role="banner" class="fr-header">',
        '<main role="main" id="content">',
        '<footer class="fr-footer fr-mt-4w" role="contentinfo">'
    ]
    
    for element in essential_elements:
        # Rechercher l'élément avec une tolérance pour les espaces et retours à la ligne
        assert element.replace(' ', '').replace('\n', '') in content.replace(' ', '').replace('\n', ''), f"Le template index.html doit inclure l'élément HTML DSFR '{element}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
