#!/bin/sh

strip-hints $1 | python -c "
import ast, sys, re

def remove_docstrings(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                node.body[0].value.s = ''
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            node.value.s = ''
    return ast.unparse(tree)

def add_selective_blank_lines(code):
    # Split into lines
    lines = code.split('\\n')
    result = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        result.append(lines[i])
        
        # Add blank line before for loops, if statements, function defs, and class defs
        if i + 1 < len(lines) and any(lines[i+1].strip().startswith(keyword) for keyword in ['for ', 'if ', 'def ', 'class ']):
            result.append('')
            
        # Add blank line after closing a block (if there's indentation change)
        if i + 1 < len(lines) and lines[i].strip() != '' and i > 0:
            curr_indent = len(lines[i]) - len(lines[i].lstrip())
            next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            
            # If indentation decreases (block ends) and next line isn't blank
            if next_indent < curr_indent and lines[i+1].strip() != '':
                result.append('')
        
        i += 1
    
    return '\\n'.join(result)

source = sys.stdin.read()
result = remove_docstrings(source)
result = re.sub(r'#.*$', '', result, flags=re.MULTILINE)

# Custom spacing
result = add_selective_blank_lines(result)

print(result)
" | black -