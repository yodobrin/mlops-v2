# mlops-v2

> This repository was "forked" from this solution [accelerator](https://github.com/Azure/mlops-v2).

Thinking about MLOps? This repo is a good starting point to get you started with MLOps. The first step is to understand the moving parts of MLOps.
The approach currently leverages few key areas:

- Azure Machine Learning (AML) CLI V2 (az ml) - see our formal docs [here](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli?tabs=public)

- DevOps pipeline or GitHub actions

- AML CLI (V2) YAML schema - see our [docs](https://learn.microsoft.com/azure/machine-learning/reference-yaml-overview)

## Prerequisites & Setup

Cloning or forking this repo.

### 0 - Create Resources

The following command will create a resource group with all required resources. The resource group will be named `rg-amlv2-<prefix>-<postfix>` and the resources will be named `<prefix>-<postfix>-<env>-<resource>`.

Once the repo is cloned you can run the following command to create the resources, alternatively you could use existing azure machine learning workspace.


> this command needs to run from the infrastructure folder

```azurecli

az deployment sub create --name <deployment-name> --location <location-of-the-service> --template-file main.bicep --parameters prefix=ydamlv2 postfix=yd2023 env=dev

```

Note for later use the following values:

- resource_group: `<The resource group of aml>`

- workspace_name: `<Azure machine learning workspace name>`

### 0.1 - CLI Setup / Update

> Run the following commands from the main folder. It is recommended to keep your az extensions up to date.

```azurecli
set -e # fail on error
python -m pip install -U --force-reinstall pip pip install azure-cli==2.35
az version
az extension add -n ml -y
az extension update -n ml
az extension list

```

### 1 - Connect to the workspace

The following commands connects your current session to the aml workspace created in the previous step, you could connect to an existing workspace as well, provide the name and the resource group of the workspace. Ensure that you are connected to the correct subscription.

```azurecli

az login
az account set --subscription <subscription id>
az account show
```

```azurecli
az configure --defaults group=<resource group name> workspace=<workspace name>
```

### 2 - Creating / Register an environment

You could create either a docker based environment or a conda based environment.

Lets register an enviorment, run it from the main folder (docker)

```azurecli
az ml environment create --file mlops/azureml/train/train-env.yml

```

### 3 - Create a compute

Now lets create the compute (in case it was not created before) if the cluster is already created you can skip this step (it wont create a second one though)

You can find what compute is already created by running the following command:

```azurecli

az ml compute list
```

Run this to make sure the compute is created:

```azurecli
az ml compute create --name cpu-cluster  --type amlcompute  --size Standard_DS3_v2 --min-instances 0   --max-instances 4   --tier low_priority

```

### 4 - Submit a training job (prep, train, evaluate, model registeration)


Now lets run a training job, submiting the job and then opening the web ui to monitor it. (This would work on mac/linux)

```azurecli

run_id=$(az ml job create -f mlops/azureml/train/pipeline.yml --query name -o tsv)

az ml job show -n $run_id --web

```

For windows, you can use the following __powershell__ command:

```powershell

$run_id = az ml job create -f mlops/azureml/train/pipeline.yml --query name -o tsv  
az ml job show -Name $run_id --web  

```

### 5 - Deploy the model

So, we called a pipeline that perform few activities such as preprocessing, training and registering the model. Now the model is registered. The next step would be to deploy an online (not batch) model. We first create an endpoint. You should verify it does not exist before running the following command.

```azurecli

az ml online-endpoint create --name <your endpoint name> -f mlops/azureml/deploy/online/online-endpoint.yml

```

After the endpoint is created we can deploy the model to it.

```azurecli

az ml online-deployment create --name taxi-online-dp --endpoint <your endpoint name> -f mlops/azureml/deploy/online/online-deployment.yml

```

after deployment, we need to route the traffic to the new model.

```azurecli

az ml online-endpoint update --name taxi-online-sf2023dev --traffic "taxi-online-dp=100"
```

### 6 - Test the model

The scoring code is generated during the training of the model. __TBD: how to get the scoring code.__
lets test it

```cli

az ml online-endpoint invoke -n <your endpoint name> --deployment-name taxi-online-dp --request-file data/taxi-request.json
```

You could also test it directly from the web ui.

![web-ui-test](images/2023-03-22-09-20-32.png)

You could also test it using other platforms, like using the SDK or the rest api.$

In the  web ui, you can acquire the scoring uri and the api key.

![consume](images/2023-03-22-09-23-53.png)

Under the folder /tests you can use the `online.rest` file to test the endpoint. Just ensure to fill in the required `.env` file with the scoring endpoint and the api key.

```env

base_url=<URL of the online/batch end-point>
api_key=<API key>

```

## Working with multiple models

It is a common practice to have multiple models for different scenarios. Each of these models will have diffrent code, enviorment, compute, etc. In this section we will see how to work with multiple models.

### Directory structure

#### Model specific code

Your model code should be hosted under the ```data-science``` folder. The folder structure is as follows:

```bash
data-science/  
│  
├── models/  
│   ├── model1/  
│   │   ├── code/  
│   │   │   ├── train.py  
│   │   │   └── prep.py
│   │   │   └── evaluate.py
│   │   │   └── register.py 
│   │   ├── environment/  
│   │   │   └── model1_environment_conda.yml  
│   │   │   └── model1_environment_docker.yml  
│   │   └── README.md  
│   │  
│   ├── model2/  
│   │   ├── code/  
│   │   │   ├── train.py  
│   │   │   └── prep.py
│   │   │   └── evaluate.py
│   │   │   └── register.py 
│   │   ├── environment/  
│   │   │   └── model2_environment.yml  
│   │   └── README.md  
│   │  
│   └── ...  
│  
└── ...  
```

#### Model specific deployment

The deployment code should be hosted under the ```mlops/azureml``` folder. The folder structure is as follows:

```bash
mlops/
│
├── azureml/
│   ├── deploy/
│   │   ├── batch/
│   │   │   ├── model1_batch-deployment.yml
│   │   │   ├── model1_batch-endpoint.yml
│   │   │   ├── model2_batch-deployment.yml
│   │   │   ├── model2_batch-endpoint.yml
│   │   │   └── ...
│   │   │
│   │   ├── online/
│   │   │   ├── model1_online-deployment.yml
│   │   │   ├── model1_online-endpoint.yml
│   │   │   ├── model2_online-deployment.yml
│   │   │   ├── model2_online-endpoint.yml
│   │   │   └── ...
│   │   │
│   │   └── ...
│   │
│   ├── train/
│   │   ├── pipeline.yml
│   │
│   └── ...
│
└── ...
```

## Putting it all together

We have discussed the required components to build a MLOps pipeline. In this section we will see how to put it all together. We will use the taxi model as an example. The use of GitHUb actions is optional. You can use any other CI/CD tool.

There are several alternatives to start a GitHub workflow, including: 
- Push event: A workflow can be triggered when a commit is pushed to a particular branch of a repository. 

- Pull request event: A workflow can be triggered when a pull request is opened, closed, or synchronized. 

- Scheduled event: A workflow can be triggered at a specific time or on a recurring schedule. 

- Repository dispatch event: A workflow can be triggered by an external event that is sent to the repository. 

- Webhook event: A workflow can be triggered by a custom webhook that is set up to listen for specific events. 

- External trigger: A workflow can be triggered by an external service, such as a continuous integration (CI) system or a deployment tool. 

These are some of the most common alternatives to start a GitHub workflow. Each of them can be configured to suit different use cases and requirements. 

GitHub, or any other CI/CD tool, will need to be authorized to access the Azure resources. This can be done by creating a service principal. The service principal will be used to authenticate the CI/CD tool to Azure. This identity will be used to create the Azure resources and to run the Azure ML pipelines. It will need to have a `contributor` role on the resource group or the subscription, depending if you want to use this identity to create the resource group or not.

### Understanding the GitHub workflow 


The GitHub workflows in this repositoryare designed to automate the process of:

- Training a machine learning model using Azure Machine Learning service.

- Online and bacth deployment of the model using Azure Machine Learning service.

These workflows are triggered by the "workflow_dispatch" event, which means that it can be manually triggered by the user. When a workflow is triggered, it asks the user to provide few inputs for example: the name of the resource group where the Azure resources are located, the name of the Azure Machine Learning workspace, and the name of the AML compute cluster to use for training the model.  

The workflow is divided into several jobs, each of which performs a specific task, the following are example of the training workflow: 

- The "register-environment" job registers the conda environment that is needed for training the model in the Azure Machine Learning workspace. This job checks out the repository and uses a custom action called "register-environment" to register the environment. The action takes three inputs: the environment file, the name of the resource group, and the name of the workspace. 

- The "create-compute" job creates an Azure Machine Learning compute cluster with the specified name, size, and node SKU. This job also checks out the repository and uses a custom action called "create-compute" to create the cluster. The action takes several inputs, including the name of the resource group, the name of the workspace, the name of the cluster, the cluster size, and the node SKU. 

- The "run-pipeline" job runs the machine learning pipeline that trains the model. This job checks out the repository and uses a custom action called "run-pipeline" to run the pipeline. The action takes several inputs, including the name of the resource group, the name of the workspace, the credentials needed to authenticate with Azure, and the location of the pipeline YAML file. 

Overall, workflows automates the process of training and deploying a machine learning model using Azure Machine Learning service, making it easier for data scientists to deploy their models quickly and efficiently.

### Creating a service principal

```azurecli
az login
az account set --subscription <your subscription id>
az ad sp create-for-rbac --name <your service principal name> --role contributor --scopes /subscriptions/<your subscription id> --sdk-auth
```

The above command will return a json object with the service principal credentials. You will need to store these credentials in a secure location. In this example we will use GitHub secrets.

There multiple blog posts and documentation on how to create a service principal. You can find more information [here](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal).

Similarly, there are multiple documents, explaining how to create a GitHub secret. You can find more information [here](https://docs.github.com/en/actions/reference/encrypted-secrets).


### Running the pipelines

#### Preparing, Training, Model Registration

As explained previously, the first step include several inner steps, such as preparing data, training and validation and model registration.

The following diagram shows the steps involved in the training pipeline:

![pipeline run](/images/2023-04-19-09-58-24.png)

#### Online and Batch Deployment

As part of this repository, we have included a GitHub workflow that automates the process of deploying the model as a web service. Follow this document to better understand the key use cases for online and batch deployment. Please review this [document](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2) to better understand the key use cases for online and batch deployment.


