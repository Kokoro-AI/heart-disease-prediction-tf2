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
$ ./bin/start [-n <string>] [-t <tag-name>] [--sudo] [--build] [-d] [-c <command>]
```

### Tags

- **latest**	The latest release of TensorFlow CPU binary image. Default.
- **nightly**	Nightly builds of the TensorFlow image. (unstable)
version	Specify the version of the TensorFlow binary image, for example: 2.1.0
- **devel**	Nightly builds of a TensorFlow master development environment. Includes TensorFlow source code.

### Variants

> Each base tag has variants that add or change functionality:

- **\<tag\>-gpu**	The specified tag release with GPU support. (See below)
- **\<tag\>-py3**	The specified tag release with Python 3 support.
- **\<tag\>-jupyter**	The specified tag release with Jupyter (includes TensorFlow tutorial notebooks)

You can use multiple variants at once. For example, the following downloads TensorFlow release images to your machine. For example:

```sh
$ ./bin/start -n myContainer --build  # latest stable release
$ ./bin/start -n myContainer --build -t devel-gpu # nightly dev release w/ GPU support
$ ./bin/start -n myContainer --build -t latest-gpu-jupyter # latest release w/ GPU support and Jupyter
```

Once the docker container is running it will execute the contents of the /bin/run file.

You can execute

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```
to access the running container's shell.

## Datasets

This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The `target` field refers to the presence of heart disease in the patient. It is integer valued `0 = no disease` and `1 = disease`.

## Models

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/run --model <model> --mode train --config <config>
```

`<model> = df_model_v1 | df_model_v2 | tfjs`
`<config> = default | heart`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/run --model <model> --mode eval --config <config>
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
