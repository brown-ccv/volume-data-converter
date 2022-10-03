import os
from erddapy import ERDDAP
import json
import typer
from halo import Halo
from pathlib import Path

app = typer.Typer()

configuration_file_path = os.path.join(Path(__file__).absolute().parent,"config","erddap_configuration.json")


@app.command()
def erddap_query(  
    output_file_path: str = typer.Argument(
        ..., help="Location where the resuilting .raw files will be saved"
    ),
    erddap_configuration_file: str = typer.Option(
        configuration_file_path,
        help=" json file with the erddap connection settings",
    ) ):
    """
        Downloads erddap data according to the settings in the erddap_configuration.json file. Downloads the data and saves it in a NetCDF file
        Keyword arguments:
        output_file_path -- Path where the NC file will be saved. It could be a directory or a path to a file with nc extension (.nc)
        erddap_configuration_file (Optional)-- Path to the errdap_confguration file. By default uses the one installed on this package.        

    """
    
    
    print(f"configuration_file_path {erddap_configuration_file}")
    try:
        conf_file = open(erddap_configuration_file, 'r')
    except OSError:
        raise(f" Could not open/read file: {erddap_configuration_file} ")

    with conf_file:
        configuration = json.load(conf_file)
    
    erddap_connection_object = configuration["erddap_connection"]

    ## check the output folder exists
    if os.path.isdir(output_file_path):
        typer.echo(f"Ouput folder path is a directory. File-name auto assigned to: {erddap_connection_object['dataset_id']}.nc" )
        output_file_path = os.path.join(output_file_path,erddap_connection_object['dataset_id']+".nc")
    else: 
        ## check the file name format and extension
        folder_path,filename = os.path.split(output_file_path)
        if not os.path.isdir(folder_path):
            raise(f" Could not find output path: {output_file_path}. Check first argument" )
        name,ext = os.path.splitext(filename)
        if ext != '.nc':
            ext = '.nc'
        output_file_path = os.path.join(folder_path,name+ext)

    # set up connection properties
    erddap_obj = ERDDAP(
        server=erddap_connection_object["server"],
        protocol=erddap_connection_object["protocol"]
    )
    erddap_obj.dataset_id = (
        erddap_connection_object["dataset_id"]
    )

    ## initialize and retrive erddap information
    erddap_obj.griddap_initialize()
    typer.echo(f"Connect to server: {erddap_connection_object['server']}")
    typer.echo(f"Looking up dataset id : {erddap_connection_object['dataset_id']}")
   
    ## set time and variables constraints
    erddap_constrainst_object = configuration["erddap_constrainst"]
    
    check_variables = all(item in erddap_obj.variables for item in erddap_constrainst_object["variables"])
    if not check_variables:
        typer.echo(f"Verify your query variables are valid in the dataset. The available variables are: {erddap_obj.variables}")
        raise Exception("Incorrect query variables")
    
    typer.echo(f"Retrieving data from server. Please be patient, it may take a while.")

    erddap_obj.variables = erddap_constrainst_object["variables"]
    erddap_obj.constraints["time>="] = erddap_constrainst_object["time>="]
    erddap_obj.constraints["time<="] = erddap_constrainst_object["time<="]

    ## erddapy doesnt support exporting data to netcdf, but xarray does. 
    spinner = Halo(text='Downloading data', spinner='dots')
    spinner.start()

    ds = erddap_obj.to_xarray()
    ds.to_netcdf(output_file_path)
    spinner.stop()
    typer.echo(f"Data downloaded to file {output_file_path}")

def main():
    app()

# if __name__ == "__main__":
#     app()