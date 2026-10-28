# Titanic CLI demo data

`titanic_train_demo.csv` is a local copy of the public Titanic training data mirrored at:

https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

The related Kaggle competition describes its training and test split here:

https://www.kaggle.com/competitions/titanic/data

Downloaded on 14 October 2026. SHA-256 of this copy:

`4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7`

Run from the repository root:

```powershell
.\.venv\Scripts\datalens.exe analyse examples\titanic_train_demo.csv
```

The dataset is a demo input for DataLens profiling, not a model trained by this project. The current core does not predict survival.
