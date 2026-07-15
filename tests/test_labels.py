from early5d import labels


def test_composition_label_ascii():
    composition = [0.10, 0.20, 0.20, 0.30]  # Mn, Ni, Co, Cu; Fe implicit at 20 %
    elements = ["Mn", "Ni", "Co", "Cu"]
    assert labels.composition_label(composition, elements, reference="Fe", unicode=False) == "Fe20Mn10Ni20Co20Cu30"


def test_composition_label_unicode_subscripts():
    composition = [0.10, 0.20, 0.20, 0.30]
    elements = ["Mn", "Ni", "Co", "Cu"]
    label = labels.composition_label(composition, elements, reference="Fe", unicode=True)
    assert label == "Fe₂₀Mn₁₀Ni₂₀Co₂₀Cu₃₀"


def test_composition_label_rounds_to_nearest_percent():
    composition = [0.001, 0.499, 0.0, 0.5]  # Ni just short of 50 % should round down, not truncate oddly
    elements = ["Mn", "Ni", "Co", "Cu"]
    label = labels.composition_label(composition, elements, reference="Fe", unicode=False)
    assert label == "Fe0Mn0Ni50Co0Cu50"
