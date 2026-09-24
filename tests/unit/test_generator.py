from fstsp.data.generator import generate_uniform_instance


def test_generator_is_deterministic():
    a = generate_uniform_instance(5, seed=7)
    b = generate_uniform_instance(5, seed=7)
    assert (a.coords == b.coords).all()
    assert a.n == 5
