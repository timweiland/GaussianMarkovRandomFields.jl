# Contributing to GaussianMarkovRandomFields.jl

Thank you for your interest in contributing to **GaussianMarkovRandomFields.jl**!
We appreciate your help in improving and maintaining this package.
The following guidelines will help you get started.

If you are looking for help rather than looking to contribute, see
[Getting Help](#getting-help) below.

## Getting Help

**Please use the [issue tracker](https://github.com/timweiland/GaussianMarkovRandomFields.jl/issues)
for all questions and support requests.** We prefer this over private email so
that answers remain searchable and benefit other users.

- **Usage questions** ("how do I model X?", "which solver should I pick?"):
  open an issue and label it `question`. Please include a short description of
  what you are trying to model.
- **Bug reports**: see [Reporting Issues](#reporting-issues) below.
- **Feature requests**: open an issue describing your use case before writing
  code, so we can discuss the design.

Before opening an issue, it is worth skimming the
[documentation](https://timweiland.github.io/GaussianMarkovRandomFields.jl/stable),
which includes tutorials and a full API reference, and searching existing issues.

## Getting Started

1. **Fork and Clone** the repository:
   ```sh
   git clone https://github.com/timweiland/GaussianMarkovRandomFields.jl.git
   cd GaussianMarkovRandomFields.jl
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
   Pkg.test("GaussianMarkovRandomFields")
   ```

## Code Style

- Follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/).
- Use meaningful variable names and avoid excessive abbreviations.
- Format your code using [Runic](https://github.com/fredrikekre/runic):
  ```sh
  make format
  # or
  runic --inplace .
  ```
  If you use pre-commit, install hooks once with `pre-commit install`.

## Making Changes

- **Open an issue** before implementing new features to discuss your idea.
- **Document your code** with docstrings using Julia’s `@doc` format.
- **Write tests** for new functionality (see next section).
- **Ensure tests pass** before submitting your changes.

## Testing

GaussianMarkovRandomFields.jl uses `Test.jl` for unit tests. To run tests:

```julia
using Pkg
Pkg.test("GaussianMarkovRandomFields")
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

If you find a bug or have a feature request, please [open an issue](https://github.com/timweiland/GaussianMarkovRandomFields.jl/issues). When reporting bugs:
- Provide a **minimal reproducible example**.
- Include Julia and GaussianMarkovRandomFields.jl version information.
- Describe expected vs. actual behavior.

## Support and Governance

GaussianMarkovRandomFields.jl is maintained by
[Tim Weiland](https://github.com/timweiland), with contributions from the
community. All development happens in the open on GitHub.

What you can expect:

- Issues and pull requests are triaged on a best-effort basis. The package is
  developed alongside other research commitments, so please allow a couple of
  weeks for a response — and do feel free to bump a thread that has gone quiet.
- Bug reports that affect correctness take priority over feature requests.
- Breaking changes are signalled by the version number, following the Julia
  ecosystem's semantic versioning conventions, and described in the
  [release notes](https://github.com/timweiland/GaussianMarkovRandomFields.jl/releases).

The maintainer has final say over the scope and design of the package. For
anything substantial, please open an issue first so we can agree on an approach
before you invest time in an implementation.

## Citing

If you use GaussianMarkovRandomFields.jl in your research, please cite it. See
[Citing](./README.md#citing) in the README, or use the *Cite this repository*
button on GitHub, which reads [`CITATION.cff`](./CITATION.cff).

## License

By contributing, you agree that your contributions will be licensed under the same license as the repository.

Thank you for contributing to **GaussianMarkovRandomFields.jl**! 🚀


