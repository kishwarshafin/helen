# Builds a cache of binaries which can just be copied for CI
venv: venv/bin/activate
IN_VENV=. ./venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv venv --python=python3 --prompt "(helen) "
	${IN_VENV} && pip install pip --upgrade


.PHONY: install
install: venv | $(addprefix $(BINCACHEDIR)/, $(BINARIES))
	${IN_VENV} && pip install -r requirements.txt
	${IN_VENV} && INSTALL_BINARIES=1 python setup.py install

.PHONY: clean
clean:
	(${IN_VENV} && python setup.py clean) || echo "Failed to run setup.py clean"
	rm -rf venv build dist/ *.egg-info/ __pycache__ *.egg-info
	find . -name '*.pyc' -delete

.PHONY: build
build: pypi_build/bin/activate
IN_BUILD=. ./pypi_build/bin/activate
pypi_build/bin/activate:
	test -d pypi_build || virtualenv pypi_build --python=python3 --prompt "(pypi) "
	${IN_BUILD} && pip install pip --upgrade
	${IN_BUILD} && pip install --upgrade pip setuptools twine wheel readme_renderer[md]