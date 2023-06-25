#### run these commands from CMD
` mkdir mlflow `

` cd mlflow `

` code . `
#### -----------------------------------------------------------------------------------------------
#### bash terminal
` conda create -p env python=3.8 -y `

` source activate ./env `

#### -----------------------------------------------------------------------------------------------
#### Shortcut
` conda create -p env python=3.8 -y && source activate ./env `

#### -----------------------------------------------------------------------------------------------
#### Create requirements.txt file
` touch requirements_mlflow.txt `

#### -----------------------------------------------------------------------------------------------
#### Install requirements.txt file
` pip install -r requirements_mlflow.txt `

#### -----------------------------------------------------------------------------------------------
#### check all python libraries
`pip list `

#### -----------------------------------------------------------------------------------------------
#### To see Mlflow GUI
` mlflow ui `