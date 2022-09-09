python -m venv venv
activate() {
    . /venv/Scripts/activate
    echo "installing requirements to virtual environment"
    pip install -r requirements.txt
}
activate