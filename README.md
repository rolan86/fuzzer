# fuzzer
This is a package to get the survival of passengers based on RandomForest Algorithm over the Titanic dataset.
The package uses the titanic dataset which is available under the data/train.csv
The package is served through Flask REST API.

You can start up the server using the command:
python main.py

You can then request for the survival rates of passengers by the following endpoint:
http://<YOUR_SERVER_IP>:8000/api/v1/survival?name=<NAME_OF_PASSENGER>

Where Name would be the name of the passenger.
It would return you a JSON object containing the name of the passenger along with the survival of the passenger, it is considered that 0 is deceased and 1 is survived.
