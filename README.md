# mlops-v2

> This repository was "forked" from this solution [accelerator](https://github.com/Azure/mlops-v2).

Thinking about MLOps? This repo is a good starting point to get you started with MLOps. The first step is to understand the moving parts of MLOps.
The approach currently leverages few key areas:

- Azure Machine Learning (AML) CLI V2 (az ml) - see our formal docs [here](https://learn.microsoft.com/azure/machine-learning/how-to-configure-cli?tabs=public)

- DevOps pipeline or GitHub actions

- AML CLI (V2) YAML schema - see our [docs](https://learn.microsoft.com/azure/machine-learning/reference-yaml-overview)

This repository provides a starting point to get you started with MLOps. It uses Azure Machine Learning (AML) CLI V2, GitHub actions, and AML CLI YAML schema. To start, clone or fork the repo and follow the steps in the Prerequisites & Setup section. You can work with multiple models, and the directory structure and deployment code should be hosted under the `data-science` and `mlops/azureml` folders, respectively. The GitHub workflows in this repository automate the process of training and deploying a machine learning model using Azure Machine Learning service.


## Table of Contents


- [Prerequisites & Setup](#prerequisites--setup)
   - [0 - Create Resources](#0---create-resources)
   - [0.1 - CLI Setup / Update](#01---cli-setup--update)
   - [1 - Connect to the workspace](#1---connect-to-the-workspace)
   - [2 - Creating / Register an environment](#2---creating--register-an-environment)
   - [3 - Create a compute](#3---create-a-compute)
   - [4 - Submit a training job (prep, train, evaluate, model registeration)](#4---submit-a-training-job-prep-train-evaluate-model-registeration)
   - [5 - Deploy the model](#5---deploy-the-model)
   - [6 - Test the model](#6---test-the-model)
- [Working with multiple models](#working-with-multiple-models)
- [Putting it all together](#putting-it-all-together)
    - [Understanding the GitHub workflow](#understanding-the-github-workflow)
    - [Creating a service principal](#creating-a-service-principal)
    - [Running the pipelines](#running-the-pipelines)
- [CLI vs. SDK](#cli-vs-sdk)


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
│   │   │   └── dependencies.yml  
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


### Creating a service principal

As GitHub action would need to perform actions on Azure, it will need to be authorized to do so. This can be done by creating a service principal. The service principal will be used to authenticate the GitHub action to Azure. This identity will be used to create the Azure resources and to run the Azure ML pipelines. It will need to have a `contributor` role on the resource group or the subscription, depending if you want to use this identity to create the resource group or not.

```azurecli
az login
az account set --subscription <your subscription id>
az ad sp create-for-rbac --name <your service principal name> --role contributor --scopes /subscriptions/<your subscription id> --sdk-auth
```

The above command will return a json object with the service principal credentials. You will need to store these credentials in a secure location. In this example we will use GitHub secrets.

There multiple blog posts and documentation on how to create a service principal. You can find more information [here](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal).

Similarly, there are multiple documents, explaining how to create a GitHub secret. You can find more information [here](https://docs.github.com/en/actions/reference/encrypted-secrets).


### Understanding the GitHub workflow 

The GitHub workflows in this repository are designed to automate the process of:

- Training a machine learning model using Azure Machine Learning service.

- Online and bacth deployment of the model using Azure Machine Learning service.

This repo includes two variation of workflows, one type is leveraging composite workflow, and the other type using workflow steps. It would be up-to-you to decide which type to take.

The composite workflows would be more suitable for larger teams, with established Ops team.

Since both types of workflows are using the same set of actions, we will only discuss the composite workflow in this section.

#### Training workflow:  ```deploy-model-training-pipeline-classical.yml``` 

Is designed to deploy a machine learning model training pipeline on Microsoft Azure. In summary, this workflow automates the process of setting up the environment, creating a compute cluster, and running a machine learning model training pipeline on Azure.

The workflow has three main jobs:  

- Register environment: This job checks out the code repository and registers the environment using a YAML file called "train-env.yml". This environment is required for the pipeline to run successfully. The job creates an environment in the Azure Machine Learning workspace specified in the input and the environment is used for training the machine learning model. 

- Create compute: This job checks out the code repository and creates a compute cluster in the same Azure Machine Learning workspace specified in the input. This cluster is used to train the machine learning model. The job uses the resource group and AML workspace name provided in the input to create the compute cluster. 

- Run pipeline: This job checks out the code repository and runs the machine learning model training pipeline using a YAML file called "pipeline.yml". The pipeline is executed on the compute cluster created in the previous job. The job uses the resource group and AML workspace name provided in the input to run the pipeline. 

Overall, the workflow automates the process of setting up the environment, creating a compute cluster, and running a machine learning model training pipeline on Microsoft Azure. This allows for a faster and more streamlined process of deploying a machine learning model on Azure.


#### Deploy to an online endpoint: ```deploy-online-endpoint-pipeline-classical.yml```

Is designed to deploy a machine learning model as online endpoint. The workflow is triggered by a manual event, which prompts the user to input several parameters including the name of the resource group and Azure Machine Learning workspace, the deployment and endpoint files, and the name of the endpoint and deployment.  

The workflow consists of three jobs, each of which runs on a Ubuntu virtual machine. The first job, "create-endpoint", checks out the repository and creates the endpoint using a custom GitHub action. The second job, "create-deployment", checks out the repository and creates the deployment using another custom GitHub action. The third job, "allocate-traffic", checks out the repository and allocates traffic to the deployment using a third custom GitHub action.  

Each job requires the output of the previous job to run, so the "create-deployment" job needs the "create-endpoint" job to finish first, and the "allocate-traffic" job needs the "create-deployment" job to finish first.  

Overall, this workflow automates the process of deploying a machine learning model as an online endpoint on Azure.

#### Deploy to a batch endpoint: ```deploy-batch-endpoint-pipeline-classical.yml```

The inputs for the workflow are the name of the resource group, AML workspace, AML Batch Compute cluster name, deployment file, endpoint file, endpoint name, endpoint type, and deployment name. 

The workflow has three jobs: 
create-compute: This job checks out the code repository and creates a compute cluster in the same Azure Machine Learning workspace specified in the input. This compute cluster is used to run the batch endpoint pipeline. The job uses the resource group and AML workspace name provided in the input to create the compute cluster. 
create-endpoint: This job checks out the code repository and creates an endpoint in the same Azure Machine Learning workspace specified in the input. This endpoint is used to host the batch scoring service. The job uses the resource group, AML workspace name, and endpoint details provided in the input to create the endpoint. 
create-deployment: This job checks out the code repository and creates a deployment in the same Azure Machine Learning workspace specified in the input. This deployment is used to deploy the batch scoring service to the endpoint created in the previous job. The job uses the resource group, AML workspace name, endpoint details, and deployment details provided in the input to create the deployment. 

Overall, the workflow automates the process of creating a compute cluster, an endpoint, and a deployment to deploy a batch endpoint pipeline on Microsoft Azure. This allows for a faster and more streamlined process of deploying a batch endpoint pipeline on Azure.



### Running the pipelines

#### Preparing, Training, Model Registration

As explained previously, the first step include several inner steps, such as preparing data, training and validation and model registration.

The following diagram shows the steps involved in the training pipeline:

![pipeline run](/images/2023-04-19-09-58-24.png)

#### Online and Batch Deployment

As part of this repository, we have included a GitHub workflow that automates the process of deploying the model as a web service. Follow this document to better understand the key use cases for online and batch deployment. Please review this [document](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2) to better understand the key use cases for online and batch deployment.

![online deployment](/images/2023-04-30-17-22-49.png)

![batch deployment](/images/2023-04-30-17-24-16.png)




## CLI vs SDK

As these alternatives are available, you may be wondering which one to use. The following table summarizes the main differences between the v2 CLI and the Python SDK in Azure Machine Learning (AML):

| Aspect | v2 CLI | Python SDK |
| --- | --- | --- |
| Interface | Command line | Code |
| Flexibility | Limited | High |
| Learning curve | Low | High |
| Integration with other tools | Limited | High |
| Compatibility | Windows, macOS, Linux | Python environment required |


- [CLI](https://docs.microsoft.com/en-us/cli/azure/ext/azure-cli-ml/ml?view=azure-cli-latest)

- [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

The main differences between using the v2 CLI and the Python SDK in Azure Machine Learning (AML) are:

Command line interface vs. code: The v2 CLI provides a command line interface to interact with AML, while the Python SDK allows you to use code to interact with AML.

Flexibility and customizability: With the Python SDK, you have more flexibility and customizability compared to the v2 CLI. You can write your own code and use various Python libraries and frameworks to implement your machine learning workflows. On the other hand, the v2 CLI provides a set of pre-defined commands that you can use to execute common tasks in AML.

Learning curve: The learning curve for the v2 CLI is generally considered to be lower compared to the Python SDK. This is because the v2 CLI is designed to be intuitive and easy to use, with a simple set of commands that can be learned quickly. The Python SDK, on the other hand, requires a higher level of expertise in Python programming.

Integration with other tools: The Python SDK can be integrated with other tools and frameworks, such as Jupyter notebooks, PyCharm, Visual Studio, and many others. This allows you to seamlessly integrate your AML workflows with your existing development environment. The v2 CLI does not have the same level of integration with other tools.

Compatibility: The v2 CLI is compatible with Windows, macOS, and Linux operating systems, while the Python SDK requires a Python environment to be set up on your machine.

In summary, the choice between using the v2 CLI or the Python SDK in Azure Machine Learning depends on your specific needs and level of expertise in Python programming. If you prefer a simple and easy-to-use command line interface, then the v2 CLI may be the best choice for you. If you need more flexibility and customizability, or if you want to integrate your AML workflows with other tools and frameworks, then the Python SDK may be a better fit.
