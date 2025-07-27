#!/usr/bin/python3
"""
Example showing how to make Flash Attention optional via command line
"""

# This would go in run.py:

# 1. Add argument (around line 50 with other arguments):
# argp.add_argument("--no_flash_attention", action="store_true", 
#                   help="Disable Flash Attention and use original attention")

# 2. Modify the model creation section (around line 128):
# if not args.no_flash_attention:
#     # Convert to Flash Attention for faster training
#     gpt = replace_attention_with_flash_attention(gpt)
#     print("Using Flash Attention for faster training")
# else:
#     print("Using original attention implementation")

# Then you could run:
# python run.py inequality_finetune ... --no_flash_attention  # Uses original
# python run.py inequality_finetune ...                       # Uses Flash (default)

print("Example: How to make Flash Attention optional")
print("=" * 60)
print("\n1. Add command line argument:")
print('argp.add_argument("--no_flash_attention", action="store_true",')
print('                  help="Disable Flash Attention and use original attention")')

print("\n2. Make conversion conditional:")
print("if not args.no_flash_attention:")
print("    gpt = replace_attention_with_flash_attention(gpt)")
print("    print('Using Flash Attention for faster training')")
print("else:")
print("    print('Using original attention implementation')")

print("\n3. Usage:")
print("python run.py inequality_finetune ...                    # Flash Attention (default)")
print("python run.py inequality_finetune ... --no_flash_attention  # Original attention")

print("\nNote: Flash Attention is recommended for better performance!")