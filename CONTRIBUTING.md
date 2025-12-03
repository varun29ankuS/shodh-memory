# Contributing to Shodh-Memory

Thank you for your interest in contributing to Shodh-Memory! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Documentation](#documentation)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/shodh-memory
   cd shodh-memory
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/varun29ankuS/shodh-memory
   ```

## Development Setup

### Prerequisites

- **Rust 1.70 or higher** - [Install Rust](https://www.rustup.rs/)
- **Git** - [Install Git](https://git-scm.com/)
- **Optional**: Docker for testing containers

### Building from Source

```bash
# Build debug version
cargo build

# Build release version
cargo build --release

# Run the server
cargo run

# Run with debug logging
RUST_LOG=debug cargo run
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run tests with coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin --verbose
```

### Code Quality Checks

```bash
# Format code
cargo fmt

# Check formatting without making changes
cargo fmt -- --check

# Run clippy (linter)
cargo clippy

# Run clippy with all features
cargo clippy --all-targets --all-features

# Check for common mistakes
cargo clippy -- -D warnings
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-endpoint` - New features
- `fix/memory-leak-issue` - Bug fixes
- `docs/update-readme` - Documentation updates
- `refactor/optimize-storage` - Code refactoring
- `test/add-integration-tests` - Test additions

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(api): add multimodal search endpoint

Implement new /api/search/multimodal endpoint that supports
five retrieval modes: similarity, temporal, causal, associative,
and hybrid.

Closes #123
```

## Testing

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Aim for >60% code coverage
- Tests should be clear and well-documented

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        // Arrange
        let input = setup_test_data();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected);
    }
}
```

### Integration Tests

Place integration tests in `tests/` directory:

```rust
// tests/api_integration_test.rs
use shodh_memory::*;

#[tokio::test]
async fn test_api_endpoint() {
    // Test implementation
}
```

## Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "feat: add awesome feature"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

5. **Create Pull Request** on GitHub

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what and why
- **Link issues**: Use "Closes #123" or "Fixes #456"
- **Tests**: Include test results
- **Documentation**: Update relevant docs
- **Screenshots**: For UI changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated
- [ ] All tests pass
```

## Code Style

### Rust Style Guide

Follow the [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/):

- Use `cargo fmt` for automatic formatting
- Maximum line length: 100 characters
- Use meaningful variable names
- Add comments for complex logic
- Use `#[allow(dead_code)]` sparingly

### Code Organization

```rust
// Imports at top
use std::collections::HashMap;

// Constants
const MAX_SIZE: usize = 100;

// Type definitions
pub struct MyStruct { ... }

// Implementation
impl MyStruct { ... }

// Tests at bottom
#[cfg(test)]
mod tests { ... }
```

### Documentation Comments

```rust
/// Brief one-line description
///
/// Longer description with more details.
///
/// # Arguments
///
/// * `param` - Description of parameter
///
/// # Returns
///
/// Description of return value
///
/// # Examples
///
/// ```
/// let result = function(param);
/// ```
pub fn function(param: Type) -> ReturnType {
    // Implementation
}
```

## Documentation

### Updating Documentation

- Update README.md for user-facing changes
- Update DOCUMENTATION.md for technical details
- Add inline code comments for complex logic
- Update examples for new features

### Documentation Standards

- Clear, concise language
- Code examples for all public APIs
- Link to related documentation
- Keep documentation in sync with code

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] Python SDK development
- [ ] Performance benchmarks
- [ ] Integration tests
- [ ] Example projects (LangChain, LlamaIndex)
- [ ] Documentation improvements

### Medium Priority
- [ ] JavaScript/TypeScript SDK
- [ ] Web dashboard
- [ ] Additional storage backends
- [ ] Performance optimizations
- [ ] Error handling improvements

### Low Priority
- [ ] Distributed mode
- [ ] GraphQL API
- [ ] Kubernetes operator
- [ ] Monitoring dashboard

## Getting Help

- **Discord**: Join our [Discord server](https://discord.gg/shodh-memory)
- **GitHub Issues**: [Create an issue](https://github.com/varun29ankuS/shodh-memory/issues)
- **Email**: 29.varuns@gmail.com

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing to Shodh-Memory! 🧠
