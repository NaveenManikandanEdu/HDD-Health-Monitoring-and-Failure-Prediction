import os

# ===== CONFIG =====
MODULES = [
    "controller",
    "data",
    "ml",
    "simulator",
    "web_app"
]

# Add .csv here 
IGNORE_EXTENSIONS = {
    ".pkl", ".parquet", ".db", ".sqlite3",
    ".pdf", ".joblib", ".h5", ".pt", ".pth",
    ".pyc", ".csv"
}

IGNORE_FOLDERS = {
    "__pycache__", ".git", ".venv",
    "venv", "env", "conda"
}


# ===== TREE GENERATOR =====
def generate_tree(root_dir):
    lines = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

        level = root.replace(root_dir, "").count(os.sep)
        indent = "│   " * level
        lines.append(f"{indent}{os.path.basename(root)}/")

        sub_indent = "│   " * (level + 1)
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() not in IGNORE_EXTENSIONS:
                lines.append(f"{sub_indent}{file}")

    return "\n".join(lines)


# ===== FILE DUMPER =====
def dump_files(root_dir, output):
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in IGNORE_EXTENSIONS:
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                output.write("\n" + "=" * 80 + "\n")
                output.write(f"FILE: {file_path}\n")
                output.write("=" * 80 + "\n\n")
                output.write(content + "\n")

            except Exception:
                continue


# ===== EXPORT FUNCTION =====
def export_module(module_name):
    if not os.path.exists(module_name):
        print(f"⚠ Skipping {module_name} (not found)")
        return

    output_filename = f"{module_name}_snapshot.txt"

    with open(output_filename, "w", encoding="utf-8") as output:
        output.write(f"{module_name.upper()} DIRECTORY TREE\n")
        output.write("=" * 80 + "\n\n")
        output.write(generate_tree(module_name))
        output.write("\n\n")
        output.write("=" * 80 + "\n")
        output.write("FILE CONTENTS\n")
        output.write("=" * 80 + "\n\n")

        dump_files(module_name, output)

    print(f" Created: {output_filename}")


# ===== MAIN =====
if __name__ == "__main__":
    for module in MODULES:
        export_module(module)

    print("\n All module snapshots created successfully (CSV files ignored).")