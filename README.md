# hearth-disease-prediction

## Content

- [Quickstart](#quickstart)
- [Datasets](#datasets)
- [Models](#models)
  - [Dense Features Model](#df_model)
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

to access the rudf_modeling container's shell.

## Datasets

**PUT INFO ABOUT THE USED DATASETS HERE!**

## Models

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/execute --model <model> --mode train --config <config>
```

`<model> = df_model_v1 | df_model_v2 | tfjs`
`<config> = default | heart`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/execute --model <model> --mode eval --config <config>
```

`<model> = df_model_v1 | df_model_v2 | tfjs`
`<config> = default | heart`

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
<dataset> = heart | ?
<model> = <model> = df_model_v1 | df_model_v2 | tfjs
```

To run TensorBoard, use the following command:

```sh
$ tensorboard --logdir=./results/<dataset>/<model>/summaries
```
