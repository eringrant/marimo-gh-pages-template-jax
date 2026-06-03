# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     # Intel macOS: jaxlib >=0.5 ships no x86_64-macOS wheel, so cap at 0.4.x.
#     "jax<0.5 ; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     "jaxlib<0.5 ; sys_platform == 'darwin' and platform_machine == 'x86_64'",
#     # Everywhere else (Linux/CI, Apple Silicon, Windows): latest JAX.
#     "jax ; sys_platform != 'darwin' or platform_machine != 'x86_64'",
# ]
# ///
import marimo

app = marimo.App()

with app.setup:
    import marimo as mo


@app.cell
def _():
    mo.md(
        r"""
        # Static JAX export: autodiff at build time

        Real Python **JAX** cannot run in the browser (`jaxlib` has no Pyodide
        wheel), so this notebook is exported with `marimo export html`: it is
        **executed at build time** and its outputs are baked into static HTML.
        JAX runs wherever the build runs (CI or your machine); the deployed page
        is a non-interactive snapshot.

        The script header caps `jax<0.5` **only on Intel macOS** (where jaxlib
        >=0.5 ships no x86_64 wheel) and uses the latest JAX everywhere else.
        """
    )
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from jax import grad, vmap

    xs = jnp.linspace(0.0, 2.0 * jnp.pi, 7)
    # vmap(grad(sin)) computes d/dx sin(x) = cos(x) at every point.
    dsin = vmap(grad(jnp.sin))(xs)

    rows = "\n".join(
        f"| {float(x):.3f} | {float(jnp.sin(x)):+.3f} | {float(d):+.3f} | {float(jnp.cos(x)):+.3f} |"
        for x, d in zip(xs, dsin)
    )
    mo.md(
        f"Computed by JAX **{jax.__version__}** at export time "
        f"(`vmap(grad(sin))`):  `STATIC_JAX_OK`\n\n"
        "| x | sin(x) | grad (autodiff) | cos(x) (exact) |\n"
        "|---|---|---|---|\n" + rows
    )
    return


if __name__ == "__main__":
    app.run()
