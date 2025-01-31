# Create dataset of time snapshots from initial client data

File named data.xlsx with initial data should be put in sequoia/data directory.
It should contain one table per sheet.
required names on sheets can be found in data_config.yaml file.

Run
```bash
python main.py
```
from sequoia directory.
Final dataset will be written do sequoia/data/ directory