import os
import cv2
import numpy as np
import imagesize
import math
import logging
from erddapy import ERDDAP
import json
import typer
import sys

app = typer.Typer()

configuration_file_path = "configuration.json"

@app.command()
def erddap_query( erddap_configuration_file: str = typer.Option(
        configuration_file_path,
        help=" json file with the erddap connection settings",
    ) ):
    
    try:
        conf_file = open(erddap_configuration_file, 'r')
    except OSError:
        typer.echo(" Could not open/read file: "+ erddap_configuration_file)
        sys.exit()

    with conf_file:
        configuration = json.load(conf_file)

    erddap_connection_object = configuration["erddap_connection"]
    if not erddap_connection_object:
        raise Exception("Json File is missed erddap_connection property")

    e = ERDDAP(
        server=erddap_connection_object["server"],
        protocol=erddap_connection_object["protocol"]
    )
    e.dataset_id = (
        erddap_connection_object["dataset_id"]
    )
    e.griddap_initialize()
    print(f"variables in this dataset:\n\n{e.variables}")
    print(
        f"\nconstraints of this dataset:\n\n{json.dumps(e.constraints, indent=1)}"
    )

if __name__ == "__main__":
    app()