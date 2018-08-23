# feature extraction

Welcome to the feature extraction project!

This project came from previous work from @escorcia during his PhD dealing with feature extraction for lots of images.

## Setup

Let's say you wanna run our codebase. You will need the following requirements:

- Linux box, x64.

- conda.

    In case, it's your first time with conda. You can do the following:

    ```
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    ```

    __Note__: This will install miniconda3 in your $HOME folder. Edit the last part as u prefer 😉.

Once, you are ready just type:

`conda env create -n pytorch -f environment-devel.yml`

> BTW, `environment-devel.yml` lists an onvercomplete set of dependencies i.e. the requirement are much less to those mentioned there.

## How to use it?

The [notebook](https://git.corp.adobe.com/escorcia/moments-retrieval/blob/adobe/notebooks/4-feature-extraction.ipynb), adobe-internal project, outlines the steps to use this code.

## When to use it?

Whenever you need to extract a lot of conv-net features from a static repo of images. You can extract features for video by dumping their frames. As long as your application/system is fine with the following pipeline

| video-source | -> frame-extraction -> | image-source | -> feature-extraction -> | conv-net features| -> your-application

This project may be useful for you.