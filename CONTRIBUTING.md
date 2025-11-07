# Contributing Guide

Thank you for your interest in contributing to YanNian-Mol! We welcome all forms of contributions.

---

## 🌟 How to Contribute

### Reporting Bugs

If you find a bug, please:

1. Check [Issues](https://github.com/yuzhounaut/YanNian-Mol/issues) to ensure it hasn't been reported
2. Create a new Issue with:
   - Clear, descriptive title
   - Detailed problem description
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment information (Python version, OS, etc.)
   - Error logs (if applicable)

### Suggesting Features

We welcome feature suggestions:

1. Create a Feature Request Issue
2. Describe the feature's purpose and value
3. Provide use case examples
4. Discuss potential implementation approaches

### Submitting Code

#### Development Workflow

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone
   git clone https://github.com/YOUR_USERNAME/YanNian-Mol.git
   cd YanNian-Mol
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Set up development environment**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Write code**
   - Follow code standards
   - Add tests
   - Update documentation

5. **Run tests**
   ```bash
   pytest tests/ -v
   black lifespan_predictor/ tests/
   flake8 lifespan_predictor/ tests/
   ```

6. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve bug"
   ```

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub

#### Commit Message Convention

Use semantic commit messages:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `style:` Code formatting (no functional changes)
- `refactor:` Code refactoring
- `test:` Test-related changes
- `chore:` Build/tooling changes

Examples:
```
feat: add support for new fingerprint type
fix: resolve memory leak in featurizer
docs: update installation guide
test: add unit tests for preprocessing
```

---

## 📝 Code Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for formatting (line length 100)
- Use [Flake8](https://flake8.pycqa.org/) for linting
- Add type hints where appropriate

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input
        
    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

### Testing

- Add unit tests for new features
- Maintain test coverage > 80%
- Use pytest framework
- Test file naming: `test_*.py`

```python
def test_feature_name():
    """Test description."""
    # Arrange
    input_data = ...
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_value
```

---

## 🔍 Code Review

Pull Requests will be reviewed for:

- ✅ Code quality and style
- ✅ Test coverage
- ✅ Documentation completeness
- ✅ Functional correctness
- ✅ Performance impact

---

## 📚 Documentation

Documentation is equally important! You can:

- Fix typos
- Improve explanations
- Add examples
- Translate documentation

Documentation is located in the `docs/` directory and uses reStructuredText format.

---

## 🎯 Priorities

We especially welcome contributions in:

- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ Performance optimizations
- 🧪 Test enhancements
- 🌐 Internationalization support
- 🔬 Support for additional model organisms

---

## ❓ Need Help?

- 📖 Check the [documentation](docs/)
- 💬 Ask questions in [Issues](https://github.com/yuzhounaut/YanNian-Mol/issues)
- 📧 Contact maintainers

---

## 📜 Code of Conduct

### Our Pledge

To foster an open and welcoming environment, we pledge to:

- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Use of sexualized language or imagery
- Personal attacks or insulting comments
- Public or private harassment
- Publishing others' private information without permission
- Other unethical or unprofessional conduct

### Enforcement

Instances of unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

## 🙏 Thank You

Thank you for contributing to YanNian-Mol! Every contribution makes the project better.

---

<div align="center">

**Building the Future of Longevity Research Together**

</div>
