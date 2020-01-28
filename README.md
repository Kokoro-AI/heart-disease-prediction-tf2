# hearth-disease-prediction

## Content

- [Quickstart](#quickstart)
- [Datasets](#datasets)
- [Models](#models)
  - [NN](#nn)
    - [Training](#training)
    - [Evaluating](#evaluating)
- [Results](#results)

## Quickstart

To start the docker container execute the following command

```sh
$ ./bin/start [-n <string>] [-t <tag-name>] [--sudo] [--build]
```

```
<tag-name> = cpu | devel-cpu | gpu
```

For example:

```sh
$ ./bin/start -n my-container -t gpu --sudo --build
```

You can execute

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```

to access the running container's shell.

## Datasets

**PUT INFO ABOUT THE USED DATASETS HERE!**

## Models

### NN

> This is not a name, please change in the future!

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/nn --mode train --config <config>
```

`<config> = default | ?`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/nn --mode eval --config <config>
```

`<config> = default | ?`

## Results

In the `/results` directory you can find the results of a training processes using a `<model>` on a specific `<dataset>`:

```
.
├─ . . .
├─ results
│  ├─ <dataset>                            # results for an specific dataset.
│  │  ├─ <model>                           # results training a <model> on a <dataset>.
│  │  │  ├─ models                         # ".h5" files for trained models.
│  │  │  ├─ results                        # ".csv" files with the different metrics for each training period.
│  │  │  ├─ summaries                      # tensorboard summaries.
│  │  │  ├─ config                         # optional configuration files.
│  │  └─ └─ <dataset>_<model>_results.csv  # ".csv" file in which the relationships between configurations, models, results and
summaries are listed by date.
│  └─ summary.csv                          # contains the summary of all the training
└─ . . .
```

where

```
<dataset> = PUT DATASETS HERE SEPARATED BY "|" . . .
<model> = nn | ?
```

To run TensorBoard, use the following command:

```sh
$ tensorboard --logdir=./results/<dataset>/<model>/summaries
```
