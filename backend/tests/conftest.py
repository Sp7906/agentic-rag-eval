import os

# Tests always run offline and deterministic, even if a .env with Azure keys exists.
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""
