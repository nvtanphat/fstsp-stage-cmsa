from fstsp.algorithms.cmsa.age import AgeManager


def test_age_manager_disables_old_components():
    ages = AgeManager(age_limit=2)
    c = ("x", 0, 1)
    ages.mark_useful({c})
    assert c in ages.active()
    ages.adapt()
    assert c in ages.active()
    ages.adapt()
    assert c not in ages.active()
