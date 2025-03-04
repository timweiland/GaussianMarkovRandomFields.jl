# Contributing to GMRFs.jl

Thank you for your interest in contributing to **GMRFs.jl**!
We appreciate your help in improving and maintaining this package.
The following guidelines will help you get started.

## Getting Started

1. **Fork and Clone** the repository:
   ```sh
   git clone https://github.com/timweiland/GMRFs.jl.git
   cd GMRFs.jl
   ```
2. **Set up the environment**:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```
3. **Run tests** to ensure everything works:
   ```julia
   using Pkg
   Pkg.test("GMRFs")
   ```

## Code Style

- Follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/).
- Use meaningful variable names and avoid excessive abbreviations.
- Format your code using [`JuliaFormatter.jl`](https://github.com/domluna/JuliaFormatter.jl):
  ```julia
  using JuliaFormatter
  format(".")
  ```

## Making Changes

- **Open an issue** before implementing new features to discuss your idea.
- **Document your code** with docstrings using Julia’s `@doc` format.
- **Write tests** for new functionality (see next section).
- **Ensure tests pass** before submitting your changes.

## Testing

GMRFs.jl uses `Test.jl` for unit tests. To run tests:

```julia
using Pkg
Pkg.test("GMRFs")
```

When adding a new feature:
- Place test cases in the `test/` directory.
- Write small, focused tests that validate the correctness of your code.
- If applicable, add edge cases and performance benchmarks.

## Submitting a Pull Request

1. Push your changes to your fork and create a pull request (PR) against the `main` branch.
2. Ensure your PR:
   - Passes all tests.
   - Includes appropriate documentation and tests.
   - Provides a clear description of the changes.
3. Be open to feedback and revisions during the review process.

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/timweiland/GMRFs.jl/issues). When reporting bugs:
- Provide a **minimal reproducible example**.
- Include Julia and GMRFs.jl version information.
- Describe expected vs. actual behavior.

## License

By contributing, you agree that your contributions will be licensed under the same license as the repository.

Thank you for contributing to **GMRFs.jl**! 🚀



