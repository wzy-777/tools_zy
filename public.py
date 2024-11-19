import os

os.system("python -m build")
os.system("python -m twine upload dist/*")
os.system("rmdir /s/q dist")
os.system("rmdir /s/q tools_zy.egg-info")