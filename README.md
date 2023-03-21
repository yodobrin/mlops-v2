# mlops-v2
first steps in mlops


## Create Resources
The following command will create a resource group with all required resources. The resource group will be named "rg-amlv2-<prefix>-<postfix>" and the resources will be named "<prefix>-<postfix>-<env>-<resource>".

Once the repo is cloned you can run the following command to create the resources:


```azurecli
az deployment sub create --name amltestv2 --location northeurope --template-file main.bicep --parameters location=northeurope prefix=ydamlv2 postfix=yd2023 env=dev
```

