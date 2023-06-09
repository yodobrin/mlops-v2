
import os
import argparse
import traceback

# Azure ML sdk v2 imports
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.core.exceptions import ResourceExistsError
from azure.ai.ml.entities import AmlCompute


def parse_args():
    parser = argparse.ArgumentParser(description="Create/Update Compute")
    parser.add_argument("--cluster_tier", type=str, help="Cluster tier")
    parser.add_argument("--max_instances", type=str, help="Max number of instances")
    parser.add_argument("--min_instances", type=str, help="Min number of instances")
    parser.add_argument("--size", type=str, help="Size of VM to be created")    
    parser.add_argument("--cluster_name", type=str, help="Name of Cluster to create")
    parser.add_argument("--subscription_id", type=str, help="subscription id")
    parser.add_argument("--resource_group", type=str, help="resource group")
    parser.add_argument("--workspace_name", type=str, help="workspace name")

    return parser.parse_args()




def connect_to_aml(args):
    
    credential = DefaultAzureCredential()

    # Get a handle to workspace
    try:        
        kwargs = {"cloud": "AzureCloud"}
        # get a handle to the subscription
        ml_client = MLClient(credential, args.subscription_id, args.resource_group, args.workspace_name, **kwargs)

    except Exception as ex:
        print(f"Unable to authenticate to workspace: {args.workspace_name} in resource group: {args.resource_group} in subscription: {args.subscription_id} ")
        print(ex)


        # tries to connect using cli args if provided else using config.yaml
        ml_client = MLClient(
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name,
            credential=credential,
        )
    return ml_client


def main():
    """Main entry point for the script."""
    args = parse_args()
    print(args)
    ml_client = connect_to_aml(args)

    
    cluster_basic = AmlCompute(
        name=args.cluster_name,
        type="amlcompute",
        size=args.size,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        tier = args.cluster_tier,
    )
    ml_client.begin_create_or_update(cluster_basic).result()

if __name__ == "__main__":
    main()