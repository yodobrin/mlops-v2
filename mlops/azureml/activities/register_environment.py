# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from azure.ai.ml.entities import Environment, BuildContext

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

import json

def parse_args():
    parser = argparse.ArgumentParser(description="Register Environment")
    parser.add_argument("--environment_name", type=str, help="Name of the environment you want to register")
    parser.add_argument("--description", type=str, help="Description of the environment")
    parser.add_argument("--env_path", type=str, help="Local path of environment file(s)")
    parser.add_argument("--build_type", type=str, help="Build type: either docker or conda")
    parser.add_argument("--base_image", type=str, help="base image path", default="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04")
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

    build_type = args.build_type
    if build_type == 'docker':
        print("Using docker build context")
        environment = Environment(
            name=args.environment_name,
            build=BuildContext(path=args.env_path),
            description=args.description
        )
    elif build_type == 'conda':
        environment = Environment(
            image=args.base_image,
            conda_file=args.env_path,
            name=args.environment_name,
            description=args.description
        )
    else: 
        print("Expected 'docker' or 'conda' as build type.")
        print(ex)

    ml_client.environments.create_or_update(environment)    

if __name__ == "__main__":
    main()