all: user_manual

user_manual:
	pdflatex user_manual
	bibtex user_manual
	pdflatex user_manual
	pdflatex user_manual

html:
	cd sphinx && make html

docs: user_manual html

clean:
	rm -f *.aux
	rm -f *.dvi
	rm -f *.log
	rm -f *.toc
	rm -f *.blg
	rm -f *.bbl
	rm -f *.out
	rm -f *.pdf
	cd sphinx && make clean
