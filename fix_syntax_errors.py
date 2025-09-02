#!/usr/bin/env python3
"""
Fix Specific Syntax Errors

This script manually fixes the remaining syntax errors in the 4 problematic files.
"""

from pathlib import Path
import ast


class SyntaxErrorFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.fixes_applied = []
    
    def fix_hybrid_retriever(self):
        """Fix hybrid_retriever.py syntax errors"""
        file_path = self.project_root / "retrieval_engine/hybrid_retriever.py"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            fixed_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Look for try statements without proper indentation or content
                if 'try:' in line and i + 1 < len(lines):
                    fixed_lines.append(line)
                    
                    # Check if next line is properly indented
                    next_i = i + 1
                    while next_i < len(lines) and lines[next_i].strip() == '':
                        fixed_lines.append(lines[next_i])
                        next_i += 1
                    
                    if next_i < len(lines):
                        next_line = lines[next_i]
                        # If next line is not indented or is except/finally, add pass
                        if (not next_line.startswith('    ') and next_line.strip() != '') or next_line.strip().startswith('except') or next_line.strip().startswith('finally'):
                            fixed_lines.append('    pass  # Added to fix syntax')
                    else:
                        fixed_lines.append('    pass  # Added to fix syntax')
                    
                    i = next_i if next_i < len(lines) else i + 1
                    continue
                
                fixed_lines.append(line)
                i += 1
            
            fixed_content = '\n'.join(fixed_lines)
            
            # Test if the fixed content parses correctly
            try:
                ast.parse(fixed_content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"✓ Fixed syntax in retrieval_engine/hybrid_retriever.py")
                self.fixes_applied.append("Fixed hybrid_retriever.py")
                return True
            except SyntaxError as e:
                print(f"✗ Still has syntax error: {e}")
                return False
                
        except Exception as e:
            print(f"Error processing hybrid_retriever.py: {e}")
            return False
    
    def fix_reranker(self):
        """Fix reranker.py syntax errors"""
        file_path = self.project_root / "semantic_reranking/reranker.py"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            fixed_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                if 'try:' in line and i + 1 < len(lines):
                    fixed_lines.append(line)
                    
                    next_i = i + 1
                    while next_i < len(lines) and lines[next_i].strip() == '':
                        fixed_lines.append(lines[next_i])
                        next_i += 1
                    
                    if next_i < len(lines):
                        next_line = lines[next_i]
                        if (not next_line.startswith('    ') and next_line.strip() != '') or next_line.strip().startswith('except') or next_line.strip().startswith('finally'):
                            fixed_lines.append('    pass  # Added to fix syntax')
                    else:
                        fixed_lines.append('    pass  # Added to fix syntax')
                    
                    i = next_i if next_i < len(lines) else i + 1
                    continue
                
                fixed_lines.append(line)
                i += 1
            
            fixed_content = '\n'.join(fixed_lines)
            
            try:
                ast.parse(fixed_content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"✓ Fixed syntax in semantic_reranking/reranker.py")
                self.fixes_applied.append("Fixed reranker.py")
                return True
            except SyntaxError as e:
                print(f"✗ Still has syntax error: {e}")
                return False
                
        except Exception as e:
            print(f"Error processing reranker.py: {e}")
            return False
    
    def fix_retrieval_enhancer(self):
        """Fix retrieval_enhancer.py syntax errors"""
        file_path = self.project_root / "canonical_flow/mathematical_enhancers/retrieval_enhancer.py"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            fixed_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                if 'try:' in line and i + 1 < len(lines):
                    fixed_lines.append(line)
                    
                    next_i = i + 1
                    while next_i < len(lines) and lines[next_i].strip() == '':
                        fixed_lines.append(lines[next_i])
                        next_i += 1
                    
                    if next_i < len(lines):
                        next_line = lines[next_i]
                        if (not next_line.startswith('    ') and next_line.strip() != '') or next_line.strip().startswith('except') or next_line.strip().startswith('finally'):
                            fixed_lines.append('    pass  # Added to fix syntax')
                    else:
                        fixed_lines.append('    pass  # Added to fix syntax')
                    
                    i = next_i if next_i < len(lines) else i + 1
                    continue
                
                fixed_lines.append(line)
                i += 1
            
            fixed_content = '\n'.join(fixed_lines)
            
            try:
                ast.parse(fixed_content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"✓ Fixed syntax in canonical_flow/mathematical_enhancers/retrieval_enhancer.py")
                self.fixes_applied.append("Fixed retrieval_enhancer.py")
                return True
            except SyntaxError as e:
                print(f"✗ Still has syntax error: {e}")
                return False
                
        except Exception as e:
            print(f"Error processing retrieval_enhancer.py: {e}")
            return False
    
    def fix_hyperbolic_tensor_networks(self):
        """Fix hyperbolic_tensor_networks.py syntax errors"""
        file_path = self.project_root / "canonical_flow/mathematical_enhancers/hyperbolic_tensor_networks.py"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            fixed_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                if 'try:' in line and i + 1 < len(lines):
                    fixed_lines.append(line)
                    
                    next_i = i + 1
                    while next_i < len(lines) and lines[next_i].strip() == '':
                        fixed_lines.append(lines[next_i])
                        next_i += 1
                    
                    if next_i < len(lines):
                        next_line = lines[next_i]
                        if (not next_line.startswith('    ') and next_line.strip() != '') or next_line.strip().startswith('except') or next_line.strip().startswith('finally'):
                            fixed_lines.append('    pass  # Added to fix syntax')
                    else:
                        fixed_lines.append('    pass  # Added to fix syntax')
                    
                    i = next_i if next_i < len(lines) else i + 1
                    continue
                
                fixed_lines.append(line)
                i += 1
            
            fixed_content = '\n'.join(fixed_lines)
            
            try:
                ast.parse(fixed_content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"✓ Fixed syntax in canonical_flow/mathematical_enhancers/hyperbolic_tensor_networks.py")
                self.fixes_applied.append("Fixed hyperbolic_tensor_networks.py")
                return True
            except SyntaxError as e:
                print(f"✗ Still has syntax error: {e}")
                return False
                
        except Exception as e:
            print(f"Error processing hyperbolic_tensor_networks.py: {e}")
            return False
    
    def run_all_fixes(self):
        """Run all syntax fixes"""
        print("="*60)
        print("FIXING SPECIFIC SYNTAX ERRORS")
        print("="*60)
        
        self.fix_hybrid_retriever()
        self.fix_reranker()
        self.fix_retrieval_enhancer()
        self.fix_hyperbolic_tensor_networks()
        
        print("\n" + "="*60)
        print("SYNTAX FIXES SUMMARY:")
        print("="*60)
        for fix in self.fixes_applied:
            print(f"✓ {fix}")
        
        print(f"\nTotal syntax fixes applied: {len(self.fixes_applied)}")


def main():
    fixer = SyntaxErrorFixer()
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()