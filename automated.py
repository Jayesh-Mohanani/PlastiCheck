import nbformat

def notebook_to_py(nb_path, py_path):
    nb = nbformat.read(open(nb_path, encoding='utf-8'), as_version=4)
    with open(py_path, 'w', encoding='utf-8') as f:
        for cell in nb.cells:
            if cell.cell_type == 'code':
                f.write(cell.source + '\n\n')
# Usage:
notebook_to_py('../SummerProject.ipynb', 'models/predictor.py')