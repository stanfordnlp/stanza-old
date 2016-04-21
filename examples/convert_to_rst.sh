set -x
set -e

jupyter nbconvert *.ipynb --to rst
for f in *.rst; do
  echo 'moving ${f}'
  mv $f ../docs/example.${f}
done
