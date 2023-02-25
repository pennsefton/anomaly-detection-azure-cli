# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import load_component
from azure.ai.ml import MLClient
import glob

def main():

    # Get the credential to feed to MLClient instantiation
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    
    # Get a handle to workspace
    ml_client = MLClient.from_config(credential=credential)

    # Register each component in the 'components directory'
    yaml_files = glob.glob('../component_*.yaml')
    for file in yaml_files:
        print(file)
        cur_component = load_component(source=file)
        ml_client.components.create_or_update(cur_component)


# run script
if __name__ == "__main__":

    # run main function
    main()
