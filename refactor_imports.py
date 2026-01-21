# #!/usr/bin/env python3
# """
# Script to update imports after structural refactoring.
# """

# import re
# import os
# from pathlib import Path

# # Mapping of old import paths to new import paths
# IMPORT_MAPPINGS = {
#     # Models
#     'from src.engine.models.agents.': 'from src.models.',
#     'from src.engine.models.core.': 'from src.models.',
#     'from src.engine.models.environment.': 'from src.models.',
#     'from src.engine.models.map.': 'from src.models.',
#     'from src.engine.models.sensors.': 'from src.models.',
#     'from engine.models.agents.': 'from src.models.',
#     'from engine.models.core.': 'from src.models.',
#     'from engine.models.environment.': 'from src.models.',
#     'from engine.models.map.': 'from src.models.',
#     'from engine.models.sensors.': 'from src.models.',
    
#     # Agent manager
#     'from src.engine.agent_manager.': 'from src.agents.',
#     'from engine.agent_manager.': 'from src.agents.',
    
#     # Communication
#     'from src.messaging.': 'from src.communication.',
#     'from src.rabbitmq.': 'from src.communication.',
#     'from messaging.': 'from src.communication.',
#     'from rabbitmq.': 'from src.communication.',
    
#     # Config
#     'from src.configurations.': 'from src.config.',
#     'from src.settings.': 'from src.config.',
#     'from configurations.': 'from src.config.',
#     'from settings.': 'from src.config.',
    
#     # Utils
#     'from src.logger.': 'from src.utils.',
#     'from logger.': 'from src.utils.',
# }

# def update_imports_in_file(file_path: Path) -> bool:
#     """Update imports in a single file. Returns True if file was modified."""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         original_content = content
        
#         # Apply import mappings
#         for old_pattern, new_pattern in IMPORT_MAPPINGS.items():
#             content = content.replace(old_pattern, new_pattern)
        
#         # Also handle import statements
#         for old_pattern, new_pattern in IMPORT_MAPPINGS.items():
#             # Remove 'from ' and 'import' for pattern matching
#             old_base = old_pattern.replace('from ', '').replace('import ', '')
#             new_base = new_pattern.replace('from ', '').replace('import ', '')
#             # Handle 'import X' style
#             pattern = r'import\s+' + re.escape(old_base.replace('src.', ''))
#             replacement = 'import ' + new_base.replace('src.', '')
#             content = re.sub(pattern, replacement, content)
        
#         if content != original_content:
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 f.write(content)
#             return True
#         return False
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return False

# def main():
#     src_dir = Path('src')
#     if not src_dir.exists():
#         print("src/ directory not found!")
#         return
    
#     updated_files = []
#     for py_file in src_dir.rglob('*.py'):
#         if '__pycache__' in str(py_file):
#             continue
#         if update_imports_in_file(py_file):
#             updated_files.append(py_file)
#             print(f"Updated: {py_file}")
    
#     print(f"\nTotal files updated: {len(updated_files)}")

# if __name__ == '__main__':
#     main()
