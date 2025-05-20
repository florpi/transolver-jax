import jax
from transolver import PhysicsAttention, TransolverBlock, Transolver


def test_physics_attention_shapes():
    batch_size = 2
    num_points = 100
    channels = 64
    num_slices = 16
    num_heads = 4
    dim_per_head = 32

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, num_points, channels))

    module = PhysicsAttention(
        num_slices=num_slices,
        num_heads=num_heads,
        num_channels=channels,
        dim_per_head=dim_per_head,
    )

    params = module.init(key, x)

    output = module.apply(params, x)

    assert output.shape == (
        batch_size,
        num_points,
        channels,
    ), f"Expected shape {(batch_size, num_points, channels)}, got {output.shape}"


def test_transolver_block():
    batch_size = 2
    num_points = 100
    channels = 64
    num_slices = 16
    num_heads = 4
    dim_per_head = 32
    mlp_dim = 128

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, num_points, channels))

    module = TransolverBlock(
        num_slices=num_slices,
        num_heads=num_heads,
        num_channels=channels,
        dim_per_head=dim_per_head,
        mlp_dim=mlp_dim,
    )

    params = module.init(key, x)

    output = module.apply(params, x)

    assert output.shape == (
        batch_size,
        num_points,
        channels,
    ), f"Expected shape {(batch_size, num_points, channels)}, got {output.shape}"


def test_transolver():
    batch_size = 2
    num_points = 100
    channels = 64
    num_slices = 16
    num_heads = 4
    dim_per_head = 32
    mlp_dim = 128

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, num_points, channels))

    module = Transolver(
        num_layers = 2,
        num_slices=num_slices,
        num_heads=num_heads,
        hidden_dim=64,
        num_channels=channels,
        mlp_dim=mlp_dim,
        output_dim=channels,
    )

    params = module.init(key, x)

    output = module.apply(params, x)

    assert output.shape == (
        batch_size,
        num_points,
        channels,
    ), f"Expected shape {(batch_size, num_points, channels)}, got {output.shape}"
