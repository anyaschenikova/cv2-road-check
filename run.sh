echo "create virtual env"

python -m venv .venv
source .venv/bin/activate

echo "install deps"
pip install -r requirements.txt

echo "run main script"
python main.py