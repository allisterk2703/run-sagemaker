.PHONY: git-push update-dags

MAKEFLAGS += --silent

help:  ## Show the list of available commands
	echo "All available commands:"
	grep -h -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  🔹 %-35s %s\n", $$1, $$2}'

git-push:  ## Git pipeline
	git checkout --orphan tmp-first-push
	git add .
	git commit -m "[ADD] First push"
	git branch -M tmp-first-push main
	git push --force origin main

update-dags:  ## Update DAGs
	echo "pyenv shell airflow-env"
	echo "airflow dags reserialize"
