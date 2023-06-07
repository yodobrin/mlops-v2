# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import json

def parse_args():
    parser = argparse.ArgumentParser(description="Register dataset")
    parser.add_argument("--data_name", type=str, help="Name of the data asset to register")
    parser.add_argument("--description", type=str, help="Description of the data asset to register")
    parser.add_argument("--data_type", type=str, help="type of data asset", default='uri_file')    
    parser.add_argument("--data_path", type=str, help="path of the data")
    parser.add_argument("--subscription_id", type=str, help="subscription id")
    parser.add_argument("--resource_group", type=str, help="resource group")
    parser.add_argument("--workspace_name", type=str, help="workspace name")

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    
    credential = DefaultAzureCredential()
    try:
        kwargs = {"cloud": "AzureCloud"}
        # get a handle to the subscription
        ml_client = MLClient(credential, args.subscription_id, args.resource_group, args.workspace_name, **kwargs)

    except Exception as ex:
        print(f"Unable to authenticate to workspace: {args.workspace_name} in resource group: {args.resource_group} in subscription: {args.subscription_id} ")
        print(ex)

    
    data = Data(
        path=args.data_path,
        type=args.data_type,
        description=args.description,
        name=args.data_name
    )
    
    ml_client.data.create_or_update(data)    

if __name__ == "__main__":
    main()