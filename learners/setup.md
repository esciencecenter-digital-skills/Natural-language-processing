---
title: Setup
---

## Software Setup

::::::::::::::::::::::::::::::::::::::: discussion

### Installing Python

[Python](https://python.org) is a popular language for scientific computing, and a frequent choice
for machine learning as well.
To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).

Please set up your python environment at least a day in advance of the workshop.
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

:::::::::::::::::::::::::::::::::::::::::::::::::::

## Installing the required packages

[Pip](https://pip.pypa.io/en/stable/) is the package management system built into Python.
Pip should be available in your system once you installed Python successfully.

## Jupyter Lab

We will teach using Python in [Jupyter Lab](http://jupyter.org/), a programming environment that runs in a web browser.
Jupyter Lab is compatible with Firefox, Chrome, Safari and Chromium-based browsers.
Note that Internet Explorer and Edge are *not* supported.
See the [Jupyter Lab documentation](https://jupyterlab.readthedocs.io/en/latest/getting_started/accessibility.html#compatibility-with-browsers-and-assistive-technology) for an up-to-date list of supported browsers.

To start Jupyter Lab, open a terminal (Mac/Linux) or Command Prompt (Windows) and type the command:

```shell
jupyter lab
```

## Data Sets


<!--
FIXME: place any data you want learners to use in `episodes/data` and then use
       a relative link ( [data zip file](data/lesson-data.zip) ) to provide a
       link to it, replacing the example.com link.
-->

For the episode 01: preprocessing and word embeddings (Word2Vec):

- Download the [Algemeen Dagblad from July 21 1969](https://www.delpher.nl/nl/kranten/view?coll=ddd&query=&cql%5B%5D=%28date+_gte_+%2220-07-1969%22%29&redirect=true&sortfield=date&resultscoll=dddtitel&identifier=KBPERS01:002846018:mpeg21&rowid=3) as txt file from Delpher. To do so, click on the link and navigate to the right hand side of the web page. There you'll find an icon with an arrow pointing down:

![arrow]('fig/setup_download_arrow.png')

Click on this icon and select `txt` among the downloading options


- Download Word2Vec models trained on 6 national Dutch newspaper data spanning a time period from 1950 to 1989 (Wevers, M., 2019). These models are available on [Zenodo](https://zenodo.org/records/3237380).

