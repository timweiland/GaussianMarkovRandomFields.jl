.PHONY: format test docs docs-serve clean-cov paper

format: ## Format code using Runic
	@runic --inplace .

test: ## Run full test suite
	julia --project=. -e "using Pkg; Pkg.test()"

test-cov: ## Run full test suite with coverage
	julia --project=. -e "using Pkg; Pkg.test(; coverage=true)"
	julia --project=. -e "using Coverage; coverage = process_folder(); LCOV.writefile(\"coverage-lcov.info\", coverage)"

docs: ## Generate documentation
	julia --project=docs docs/make.jl

docs-server:
	@echo "Starting VitePress docs server..."
	@cd docs && julia --project -e 'using DocumenterVitepress; DocumenterVitepress.dev_docs("build")'

clean-cov: ## Clean up coverage files
	find . -name "*.jl.*.cov" -delete

# JOSS Paper targets
paper: paper/paper.pdf ## Build JOSS paper PDF
	@open paper/paper.pdf

paper/paper.pdf: paper/paper.md paper/paper.bib paper/bernoulli_classification.pdf
	docker run --rm -v $(CURDIR)/paper:/data openjournals/inara -o pdf paper.md

paper/bernoulli_classification.pdf: paper/generate_figure.jl paper/Manifest.toml
	julia --project=paper paper/generate_figure.jl

paper/Manifest.toml: paper/Project.toml
	julia --project=paper -e 'using Pkg; Pkg.instantiate()'
