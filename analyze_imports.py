#!/usr/bin/env python3
"""
Import Dependency Analysis Script
Analyzes import relationships in the Frankenstino AI codebase
"""

import os
import ast
import sys
from pathlib import Path

def analyze_imports(filepath):
    """Analyze imports in a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filepath)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])

        return list(set(imports))
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return []

def main():
    """Main analysis function"""
    print('=== IMPORT DEPENDENCY ANALYSIS ===\n')

    # Analyze key backend files
    backend_files = [
        'backend/main.py',
        'backend/config.py',
        'backend/utils/component_factory.py',
        'backend/memory/memory_manager.py',
        'backend/llm/llm_core.py',
        'backend/llm/memory_curator.py',
        'backend/memory/memory_taxonomy.py',
        'backend/memory/memory_metrics.py',
        'backend/learning_pipeline.py'
    ]

    dependency_graph = {}

    for filepath in backend_files:
        if os.path.exists(filepath):
            imports = analyze_imports(filepath)
            dependency_graph[filepath] = imports

            print(f'{filepath}:')
            for imp in sorted(imports):
                print(f'  - {imp}')
            print()

    # Check for potential circular dependencies
    print('=== POTENTIAL CIRCULAR DEPENDENCY CHECK ===')

    # Simple check: if A imports B and B imports A
    modules = {}
    for filepath, imports in dependency_graph.items():
        module_name = filepath.replace('.py', '').replace('/', '.').replace('\\', '.')
        modules[module_name] = imports

    circular_deps = []
    for module, deps in modules.items():
        for dep in deps:
            if dep in modules and module in modules[dep]:
                circular_deps.append((module, dep))

    if circular_deps:
        print("Potential circular dependencies found:")
        for dep in circular_deps:
            print(f"  {dep[0]} ↔ {dep[1]}")
    else:
        print("No obvious circular dependencies detected.")

    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == '__main__':
    main()
