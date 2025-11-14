# CLAUDE.md - AI Assistant Guide for Projects Repository

## Repository Overview

This is a personal projects repository containing code from multiple programming languages and learning exercises. The repository serves as a collection of:
- Personal projects created as necessary tools or for experimentation
- Modified code from other sources (with proper credit)
- Code created during online learning exercises and book studies

**Primary Languages:** Python, SQL, R, Web Development (HTML/CSS)

**Repository Structure:** Projects are organized by language/technology in top-level directories

## Directory Structure

```
/home/user/Projects/
├── Python/              # Python projects and scripts
│   ├── HTML_Document_Reader/
│   ├── Kiva_Nonprofit_Analysis/
│   ├── LIfe_Expectancy_and_GDP/
│   ├── Machine_Learning/
│   │   └── Chapter_2_Housing/
│   ├── Quiver Plots/
│   ├── Roller_Coaster_Analysis/
│   ├── Star Classifier/
│   ├── Tennis_Ace_ML/
│   ├── Twitter_Classification/
│   ├── Web_Scraping/
│   └── knapsack_problem.py
├── R/                   # R programming projects
│   └── Google Data Analytics Capstone/
├── SQL/                 # SQL projects (currently minimal)
├── Web Development/     # HTML/CSS projects
│   ├── Dasmoto Project/
│   ├── DevProject/
│   └── Simple Site/
└── README.md
```

## Project Organization Patterns

### Python Projects

**Structure:**
- Each project typically has its own subdirectory
- Projects contain a mix of `.py` scripts and `.ipynb` Jupyter notebooks
- Most projects include a `README.md` explaining the project purpose

**Common Project Types:**
1. **Data Analysis & Visualization** - Projects using pandas, matplotlib, seaborn
   - Examples: Kiva_Nonprofit_Analysis, LIfe_Expectancy_and_GDP, Roller_Coaster_Analysis

2. **Machine Learning** - ML model development and analysis
   - Examples: Tennis_Ace_ML, Star Classifier, Twitter_Classification

3. **Data Processing Tools** - Custom utilities for data extraction/transformation
   - Example: HTML_Document_Reader

4. **Algorithm Implementations** - Standalone scripts implementing algorithms
   - Example: knapsack_problem.py

### R Projects

**Structure:**
- Projects use R Markdown (.Rmd) files for literate programming
- Include both .Rmd source and compiled .pdf outputs
- Focus on data analytics and visualization

**Example:**
- Google Data Analytics Capstone - Complete case study from Coursera certification

### Web Development Projects

**Structure:**
- Standard HTML/CSS project structure
- Resources organized in subdirectories (css/, img/)
- Simple static website projects

## Code Conventions & Style

### Python Conventions

**Class Definitions:**
- Use PascalCase for class names (e.g., `HTML_Doc_Reader`)
- Include docstrings explaining class purpose
- Type hints used for function signatures (`-> None`, `-> str`, `-> List`)
- Comments explain business logic and assumptions

**Example Pattern:**
```python
class HTML_Doc_Reader():
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir

    def get_list(self) -> List:
        # Implementation with inline comments
        pass
```

**Function Style:**
- Snake_case for function names
- Descriptive function names that clearly state purpose
- Type hints for parameters and return types
- Comments explain non-obvious logic

**Imports:**
- Standard library imports first
- Third-party libraries after (pandas, nltk, etc.)
- Commented-out utility lines (e.g., `#nltk.download()`)

**Common Libraries:**
- pandas - Data manipulation
- matplotlib/seaborn - Visualization
- nltk - Natural language processing
- scikit-learn - Machine learning
- numpy - Numerical computing

### Documentation Standards

**README Files:**
Each project should include a README.md with:
1. **Project Title/Name**
2. **Overview/Purpose** - What the project does and why
3. **Details** - Implementation specifics, algorithms used, data sources
4. **Credits** - Attribution if based on tutorials or other sources

**Example Structure:**
```markdown
# Project Name

Brief description of what was learned or accomplished.

## Overview
Detailed explanation of the project purpose and functionality.

## Details
Technical implementation details, data sources, methodology.
```

## Development Workflows

### For Data Analysis Projects

1. **Setup** - Create project directory with descriptive name
2. **Data Acquisition** - Store data in project directory or reference external sources (Kaggle, etc.)
3. **Development** - Use Jupyter notebooks for exploratory analysis
4. **Documentation** - Create README explaining purpose, data sources, insights
5. **Version Control** - Commit with descriptive messages

### For Utility Scripts

1. **Class-Based Design** - Encapsulate functionality in classes
2. **Modular Functions** - Break down tasks into focused methods
3. **Type Hints** - Add type annotations for clarity
4. **Documentation** - Comment business logic and assumptions
5. **Testing** - Include test cases or usage examples

## Key File Types

### Jupyter Notebooks (.ipynb)
- **Primary Use:** Data analysis, visualization, machine learning experiments
- **Naming:** Descriptive of project purpose (e.g., `kiva_project.ipynb`, `life_expectancy_gdp.ipynb`)
- **Location:** Within project-specific subdirectories under `Python/`

### Python Scripts (.py)
- **Primary Use:** Reusable utilities, algorithm implementations, classes
- **Naming:** Snake_case descriptive names
- **Location:** Either in project subdirectories or root of `Python/` for standalone scripts

### R Markdown (.Rmd)
- **Primary Use:** Complete data analysis reports with code, narrative, and visualizations
- **Outputs:** Compiled to PDF for presentation
- **Location:** Within `R/` project subdirectories

### HTML/CSS Files
- **Structure:** Standard web project layout with resources organized in subdirectories
- **Location:** Under `Web Development/` with project-specific folders

## AI Assistant Best Practices

### When Adding New Projects

1. **Create Appropriate Directory Structure**
   - Use descriptive project names
   - Organize under correct language directory
   - Create resources subdirectories if needed

2. **Include README.md**
   - Explain project purpose
   - Document data sources and attributions
   - Describe methodology and key insights

3. **Follow Existing Conventions**
   - Match naming patterns (PascalCase for classes, snake_case for functions)
   - Use type hints in Python
   - Add meaningful comments

### When Modifying Code

1. **Preserve Style Consistency**
   - Match existing indentation and spacing
   - Follow established naming conventions
   - Maintain comment style

2. **Update Documentation**
   - Modify README.md if functionality changes
   - Update inline comments for modified logic
   - Document new dependencies

3. **Handle Dependencies**
   - Note if new libraries are required
   - Keep commented utility lines (like `#nltk.download()`)

### When Analyzing Existing Code

1. **Check README First**
   - Understand project context and purpose
   - Identify data sources and external dependencies
   - Note any attributions or learning sources

2. **Understand Project Category**
   - Data analysis projects may be exploratory
   - Utility scripts should be production-ready
   - Learning exercises may be incomplete or experimental

3. **Respect Project State**
   - Some projects are complete (capstones, course projects)
   - Others may be in-progress or experimental
   - Check README for status indicators

## Common Patterns to Recognize

### Data Analysis Pattern
```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (often from external sources like Kaggle)
df = pd.read_csv('data.csv')

# Exploratory analysis with visualizations
# Typically done in Jupyter notebooks
```

### Utility Class Pattern
```python
class DataProcessor():
    def __init__(self, config_param) -> None:
        self.config = config_param

    def process(self) -> pd.DataFrame:
        # Encapsulated processing logic
        return result
```

### Project Organization Pattern
- Each self-contained project has its own directory
- Directory name describes the project
- README.md at project level
- Data files or notebooks within project directory

## Git Workflow

**Branch Strategy:**
- Work on feature branches prefixed with `claude/`
- Branch names include session identifiers
- Current branch: `claude/claude-md-mhywnq4uuy45oe48-016eDkHxRGZxu3XG4prLZoQM`

**Commit Messages:**
- Descriptive of changes made
- Reference project name if working within specific project

**Push Strategy:**
- Push to designated feature branch
- Use `git push -u origin <branch-name>`
- Branch must start with `claude/` prefix

## Notes for AI Assistants

### Project Context Matters
- **Learning Projects** - May have incomplete implementations or TODO sections
- **Utility Projects** - Should be functional and well-documented
- **Analysis Projects** - Focus on insights, visualization, and documentation

### Data Sources
- Many projects use external data (Kaggle, public datasets)
- Data files may not be in repository (size or licensing reasons)
- Always check README for data source information

### Dependencies
- No centralized `requirements.txt` - projects are self-contained
- Common libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk
- R projects may have their own package dependencies in .Rmd files

### Attribution
- Repository includes both original work and learning exercises
- Always preserve attribution comments
- Credit sources when present in README or comments

### File Naming
- Use descriptive names that indicate content/purpose
- Python: snake_case for scripts, PascalCase for classes
- Directories: Descriptive names with spaces where readable (e.g., "Quiver Plots")

---

**Last Updated:** 2025-11-14
**Repository Type:** Personal Learning & Project Portfolio
**Primary Owner:** cjh4232
