# Output directory for all files
$out_dir = 'pdf';

# Ensure output directory exists
system("mkdir -p pdf");

# Use XeLaTeX
$pdf_mode = 1;
$pdflatex = 'xelatex -interaction=nonstopmode -synctex=1';

# Clean up temporary files
$clean_ext = 'aux bbl blg log out toc lot lof run.xml synctex.gz fls fdb_latexmk';

# Process both files
@default_files = ('manuscript.tex', 'si.tex'); 