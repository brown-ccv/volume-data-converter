import os
import cv2
import numpy as np
import imagesize
import math
import logging
from erddapy import ERDDAP
from erddapClient import ERDDAP_Server,ERDDAP_Griddap
import json
import typer
import sys

app = typer.Typer()

configuration_file_path = "erddap_configuration.json"

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

    ## set up the erddap connection
    remoteServer = ERDDAP_Griddap(erddap_connection_object["server"],
                    datasetid=erddap_connection_object["dataset_id"])

    erddap_constrainst_object = configuration["erddap_constrainst"]
    remoteServer.addConstraint(f"time>={erddap_constrainst_object['time>=']}") \
                .addConstraint(f"time<={erddap_constrainst_object['time<=']}")
    print(remoteServer)


    # erddap_obj = ERDDAP(
    #     server=erddap_connection_object["server"],
    #     protocol=erddap_connection_object["protocol"]
    # )
    # erddap_obj.dataset_id = (
    #     erddap_connection_object["dataset_id"]
    # )

    # ## initialize and retrive erddap information
    # erddap_obj.griddap_initialize()
    # print(f"variables in this dataset:\n\n{erddap_obj.variables}")
    # print(
    #     f"\nconstraints of this dataset:\n\n{json.dumps(erddap_obj.constraints, indent=1)}"
    # )
    # ncdf_data = erddap_obj.to_ncCF()
    # print(ncdf_data)

if __name__ == "__main__":
    app()