import glob
import shutil
import os

src_files = glob.glob(r"C:\Users\dhruv\Calibre Library\Patrick Rothfuss\**\*.txt", recursive=True)


pathTo = "./data/PatrickRothfuss"

os.makedirs(pathTo, exist_ok=True)

for f in src_files:
    filename = os.path.basename(f)
    dest = os.path.join(pathTo, filename)

    if os.path.exists(dest):
        name, ext = os.path.splitext(filename)
        dest = os.path.join(pathTo, f"{name}_{hash(f)}{ext}")

    shutil.copy2(f, dest)

print(f"Copied {len(src_files)} files to {pathTo}")